"""
Based on the pseudo-code provided in the paper "Shap-enhanced counterfactual explanations forrecommendations"
and the implementation from the LXR repository: https://github.com/DeltaLabTLV/LXR
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import torch
import numpy as np
import shap
import logging
from scipy import sparse
from sklearn.cluster import KMeans
from .base_model import UserVectorExpBaseModel

class SHAP(UserVectorExpBaseModel):
    def __init__(self, rec_model, device, args, config=None):
        super(SHAP, self).__init__(rec_model, device, args, config)
        self.n_items = rec_model.n_items
        self.mode = 'implicit'
        self.n_background = config.get('n_background', config.get('n_samples', 50))
        self.shap_nsamples = config.get('nsamples', 'auto')
        self.n_clusters = config.get('n_clusters', 10)
        self.to(device)
        self.background_data_source = None
        self.item_to_cluster = None
        self.cluster_to_items = None
        self._cluster_bg = None
        self._shap_background = None
        self.train_items_pool = None
        self.prepare_data(rec_model.data_handler.train_df, n_items=self.n_items)

    def prepare_data(self, train_df, n_items=None):
        if n_items is None: n_items = self.n_items

        rows = train_df['user_id'].values
        cols = train_df['item_id'].values
        data = np.ones(len(rows))

        n_users = train_df['user_id'].max() + 1
        self.background_data_source = sparse.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
        self.train_items_pool = cols
        self._setup_clustering()

    def _setup_clustering(self):
        train_matrix = self.background_data_source.toarray()

        np.random.seed(42)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")
            warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*")
            kmeans = KMeans(n_clusters=self.n_clusters, random_state=3, n_init=10)
            self.item_clusters = kmeans.fit_predict(train_matrix.T)

        self.cluster_to_items = {}
        self.item_to_cluster = {}
        for i, cluster in enumerate(self.item_clusters):
            self.item_to_cluster[i] = int(cluster)
            if cluster not in self.cluster_to_items:
                self.cluster_to_items[cluster] = []
            self.cluster_to_items[cluster].append(i)

        for c in range(self.n_clusters):
            if c not in self.cluster_to_items:
                self.cluster_to_items[c] = []

        n_users_train = train_matrix.shape[0]
        cluster_train = np.zeros((n_users_train, self.n_clusters))
        for c in range(self.n_clusters):
            items_in_c = self.cluster_to_items[c]
            if len(items_in_c) > 0:
                cluster_train[:, c] = np.sum(train_matrix[:, items_in_c], axis=1)
        self._cluster_bg = np.where(cluster_train > 0, 1, 0).astype(np.float64)

        # Build a fixed background matrix, matching LXR workflow: [target_item, user_clusters...].
        bg_size = min(int(self.n_background), self._cluster_bg.shape[0])
        rng = np.random.RandomState(3)
        sampled_rows = rng.choice(self._cluster_bg.shape[0], size=bg_size, replace=False)
        bg_clusters = self._cluster_bg[sampled_rows]
        if self.train_items_pool is not None and len(self.train_items_pool) > 0:
            bg_items = rng.choice(self.train_items_pool, size=(bg_size, 1), replace=True).astype(np.float64)
        else:
            bg_items = rng.randint(0, self.n_items, size=(bg_size, 1)).astype(np.float64)
        self._shap_background = np.hstack([bg_items, bg_clusters])

    def _compute_shap_values(self, history_indices, target_items, user_id, level='list'):
        logging.getLogger('shap').setLevel(logging.WARNING)

        user_cluster = np.zeros(self.n_clusters)
        for idx in history_indices:
            c = self.item_to_cluster[idx]
            user_cluster[c] = 1

        background = self._shap_background

        def _predict_wrapper(batch, _cluster_to_items=self.cluster_to_items, _n_clusters=self.n_clusters,
                     _n_items=self.n_items, _device=self.device):
            items = batch[:, 0].astype(int)
            clusters = batch[:, 1:]
            n_samples_batch = batch.shape[0]

            full_mask = np.zeros((n_samples_batch, _n_items), dtype=np.float32)
            for c in range(_n_clusters):
                items_in_c = _cluster_to_items[c]
                if items_in_c:
                    full_mask[:, items_in_c] = clusters[:, c:c+1]

            mask_tensor = torch.tensor(full_mask, dtype=torch.float32, device=_device)
            user_tensor = torch.tensor([user_id], device=_device, dtype=torch.long).expand(n_samples_batch)
            mask_list = [mask_tensor[i] for i in range(n_samples_batch)]

            with torch.no_grad():
                scores = self.rec_model.predict(users=user_tensor, mask=mask_list)
                batch_indices = np.arange(n_samples_batch)
                target_scores = scores[batch_indices, items].cpu().numpy()

            return target_scores

        all_cluster_shap = []
        try:
            explainer = shap.KernelExplainer(_predict_wrapper, background)
            instances = np.hstack([
                np.array(target_items, dtype=np.float64).reshape(-1, 1),
                np.tile(user_cluster.reshape(1, -1), (len(target_items), 1))
            ])
            sv = explainer.shap_values(instances, nsamples=self.shap_nsamples, silent=True)

            if isinstance(sv, list):
                sv = sv[0]
            sv = np.array(sv)
            if sv.ndim == 1:
                sv = sv.reshape(1, -1)

            all_cluster_shap = sv[:, 1:]
        except Exception:
            print("SHAP failed, returning zeros.")
            all_cluster_shap = np.zeros((len(target_items), self.n_clusters))

        if level == 'list':
            cluster_shap = np.sum(all_cluster_shap, axis=0)
            full_scores = np.zeros(self.n_items)
            for idx in history_indices:
                full_scores[idx] = cluster_shap[self.item_to_cluster[idx]]
        else:
            cluster_shap = np.array(all_cluster_shap)
            n_targets = cluster_shap.shape[0]
            full_scores = np.zeros((n_targets, self.n_items))
            for idx in history_indices:
                full_scores[:, idx] = cluster_shap[:, self.item_to_cluster[idx]]
        return torch.tensor(full_scores, dtype=torch.float32, device=self.device)

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)

        user_interactions_np = user_interaction.cpu().numpy().reshape(1, -1)
        history_mask = user_interactions_np[0] > 0
        history_indices = np.where(history_mask)[0]

        target_items = item_ids
        if not isinstance(target_items, (list, np.ndarray)):
            target_items = [target_items]

        if len(history_indices) == 0:
            return torch.zeros(self.n_items, device=self.device)

        level = 'list' if len(target_items) > 1 else 'item'
        scores_tensor = self._compute_shap_values(history_indices, target_items, user_id, level=level)
        if scores_tensor.dim() == 2 and scores_tensor.shape[0] == 1:
            scores_tensor = scores_tensor.squeeze(0)
        
        mask = torch.zeros_like(scores_tensor)
        mask[..., history_indices] = 1.0
        scores_tensor = scores_tensor * mask
        scores_tensor += user_interaction * 1e-12
        
        return scores_tensor

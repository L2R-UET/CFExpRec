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
        self.n_clusters = config.get('n_clusters', 10)
        nsamples = config.get('nsamples', 'auto')
        self.shap_nsamples = min(2 ** self.n_clusters, 2 * self.n_clusters + 2048) if nsamples == 'auto' else nsamples
        self.to(device)
        self._shap_cache = {}
        self._prepare(rec_model.data_handler.train_df)

    def _prepare(self, train_df):
        rows, cols = train_df['user_id'].values, train_df['item_id'].values
        n_users = train_df['user_id'].max() + 1
        self._train_matrix = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(n_users, self.n_items)
        ).toarray()

        item_vectors = self._train_matrix.T
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")
            warnings.filterwarnings("ignore", message=".*Number of distinct clusters.*")
            self._item_cluster_idx = KMeans(n_clusters=self.n_clusters, n_init=10).fit_predict(item_vectors).astype(np.intp)

        membership = np.zeros((self.n_items, self.n_clusters))
        membership[np.arange(self.n_items), self._item_cluster_idx] = 1
        cluster_bg = ((self._train_matrix @ membership) > 0).astype(np.float64)

        bg_size = min(int(self.n_background), cluster_bg.shape[0])
        self._shap_background = cluster_bg[np.linspace(0, cluster_bg.shape[0] - 1, bg_size, dtype=int)]

    def _parse_shap_output(self, sv, n_targets):
        if isinstance(sv, list):
            return np.atleast_2d(np.array([np.asarray(s).reshape(-1) for s in sv]))
        sv = np.atleast_2d(np.squeeze(np.asarray(sv)))
        return sv.T if sv.shape[0] != n_targets else sv

    def _compute_shap_values(self, history_indices, target_items, user_id, level='list'):
        logging.getLogger('shap').setLevel(logging.WARNING)

        target_arr = np.array(target_items, dtype=int)
        n_targets = len(target_items)
        history_mask = np.zeros(self.n_items, dtype=np.float32)
        history_mask[history_indices] = 1.0
        user_history = self.ui_mat[user_id]
        cidx = self._item_cluster_idx

        user_cluster = np.zeros(self.n_clusters)
        user_cluster[cidx[history_indices]] = 1

        def predict_fn(batch):
            n = batch.shape[0]
            mask_t = torch.tensor(batch[:, cidx].astype(np.float32) * history_mask, dtype=torch.float32, device=self.device)
            users = torch.tensor([user_id], device=self.device, dtype=torch.long).expand(n)
            uh = user_history.unsqueeze(0).expand(n, -1)
            with torch.no_grad():
                return self.rec_model.predict(users=users, mask=uh * mask_t + (1 - uh) * (1 - mask_t))[:, target_arr].cpu().numpy()

        try:
            explainer = shap.KernelExplainer(predict_fn, self._shap_background)
            sv = explainer.shap_values(user_cluster.reshape(1, -1), nsamples=self.shap_nsamples, silent=True)
            cluster_shap = self._parse_shap_output(sv, n_targets)
        except Exception:
            cluster_shap = np.zeros((n_targets, self.n_clusters))

        scores = np.zeros(self.n_items) if level == 'list' else np.zeros((n_targets, self.n_items))
        shap_per_item = cluster_shap.sum(axis=0)[cidx[history_indices]] if level == 'list' else cluster_shap[:, cidx[history_indices]]

        if level == 'list':
            scores[history_indices] = shap_per_item
        else:
            scores[:, history_indices] = shap_per_item

        return torch.tensor(scores, dtype=torch.float32, device=self.device)

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
        history_indices = np.where(user_interaction.cpu().numpy().flatten() > 0)[0]
        target_items = item_ids if isinstance(item_ids, (list, np.ndarray)) else [item_ids]

        if len(history_indices) == 0:
            return torch.zeros(self.n_items, device=self.device)

        level = 'list' if len(target_items) > 1 else 'item'

        if level == 'item':
            cache_key = user_id
            if cache_key not in self._shap_cache:
                all_targets = self.y_pred_indices[user_id].tolist()
                self._shap_cache = {cache_key: (all_targets, self._compute_shap_values(history_indices, all_targets, user_id, level='item'))}
            targets, values = self._shap_cache[cache_key]
            idx = targets.index(target_items[0]) if target_items[0] in targets else -1
            scores_tensor = values[idx] if idx >= 0 else self._compute_shap_values(history_indices, target_items, user_id, level='item').squeeze(0)
        else:
            scores_tensor = self._compute_shap_values(history_indices, target_items, user_id, level=level)

        scores_tensor = scores_tensor.squeeze(0) if scores_tensor.dim() == 2 and scores_tensor.shape[0] == 1 else scores_tensor

        mask = torch.zeros_like(scores_tensor)
        mask[..., history_indices] = 1.0
        scores_tensor = scores_tensor * mask
        scores_tensor += user_interaction * 1e-12

        return scores_tensor
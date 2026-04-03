import torch
import torch.nn as nn
import numpy as np
import random
from tqdm import tqdm
from metrics.rec_metrics import recall_at_k, ndcg_at_k

class RecBaseModel(nn.Module):
    def __init__(self, data_handler, device, args=None, config=None, **kwargs):
        super(RecBaseModel, self).__init__()
        self.device = device
        self.args = args
        self.config = config
        self.data_handler = data_handler
        self.n_users = data_handler.n_users
        self.n_items = data_handler.n_items
        self._get_interaction_matrix()
        self._get_history_mask()

    def apply_mask(self, ui_mat, mask):
        if ui_mat.is_sparse:
            raise NotImplementedError("Masking not supported for sparse matrices.")
        return ui_mat * mask + (1 - ui_mat) * (1 - mask)

    def _get_interaction_matrix(self):
        rows = self.data_handler.train_df['user_id'].values
        cols = self.data_handler.train_df['item_id'].values
        
        indices = torch.stack([
            torch.tensor(rows, dtype=torch.long, device=self.device),
            torch.tensor(cols, dtype=torch.long, device=self.device)
        ], dim=0)
        
        values = torch.ones(len(rows), device=self.device)
        
        self.ui_mat = torch.sparse_coo_tensor(
            indices, values, 
            size=(self.n_users, self.n_items),
            device=self.device
        ).coalesce()

    def train_epoch(self, optimizer):
        self.train()
        final_loss_list = []

        if hasattr(self.data_handler.train_dataset, 'negative_sampling'):
            self.data_handler.train_dataset.negative_sampling(self.n_items)

        if self.args.epoch_pbar:
            pbar = tqdm(enumerate(self.data_handler.train_loader))
        else:
            pbar = enumerate(self.data_handler.train_loader)

        for _, batch in pbar:
            users, pos_items, neg_items = batch
            users = users.to(self.device)
            pos_items = pos_items.to(self.device)
            neg_items = neg_items.to(self.device)

            optimizer.zero_grad()
            loss = self.forward(users, pos_items, neg_items)
            loss.backward()
            optimizer.step()
            final_loss_list.append(loss.item())

        return final_loss_list

    def test_epoch(self, topk):
        self.eval()
        test_topK_recall = 0.0
        test_topK_ndcg = 0.0

        with torch.no_grad():
            if self.args.epoch_pbar:
                pbar = tqdm(enumerate(self.data_handler.test_loader))
            else:
                pbar = enumerate(self.data_handler.test_loader)

            for _, batch in pbar:
                users, gt_items = batch
                users = users.to(self.device)
                _, topk_indices = self.predict(users=users, topk=topk)
                gt_items = gt_items.to(self.device)
                test_topK_recall += recall_at_k(topk_indices, gt_items, topk).sum().item()
                test_topK_ndcg += ndcg_at_k(topk_indices, gt_items, topk).sum().item()

        n_test = len(self.data_handler.test_dataset)
        return test_topK_recall / n_test, test_topK_ndcg / n_test

    def forward(self, users, pos_items, neg_items):
        raise NotImplementedError

    def compute_scores(self, users, mask=None):
        raise NotImplementedError

    def _get_history_mask(self):
        self._history_mask = torch.zeros((self.n_users, self.n_items), device=self.device)
        self._history_mask[self.data_handler.train_df['user_id'], self.data_handler.train_df['item_id']] = float('-inf')

    def predict(self, users=None, topk=None, mask=None):
        self.eval()
        scores = self.compute_scores(users, mask)
        if isinstance(users, int) and scores.dim() > 1:
            scores = scores.squeeze(0)
        if topk != None:
            if users != None:
                history_mask = self._history_mask[users]
            else:
                history_mask = self._history_mask
            scores = scores + history_mask
            topk_indices = torch.topk(scores, min(topk, self.n_items), dim=-1).indices
            return scores, topk_indices
        return scores
    
class UserVectorRecBaseModel(RecBaseModel):
    def __init__(self, data_handler, device, args=None, config=None, **kwargs):
        super(UserVectorRecBaseModel, self).__init__(data_handler, device, args, config, **kwargs)
        
    def get_interaction_vectors(self, user_ids):
        if isinstance(user_ids, int):
            user_ids = torch.tensor([user_ids], device=self.device)
        elif isinstance(user_ids, list):
            user_ids = torch.tensor(user_ids, device=self.device)
        return self.ui_mat.index_select(0, user_ids).to_dense()

class GraphRecBaseModel(RecBaseModel):
    def __init__(self, data_handler, device, args=None, config=None, **kwargs):
        super(GraphRecBaseModel, self).__init__(data_handler, device, args, config, **kwargs)
        self.explain = kwargs.get('explain', False)
        if self.explain:
            self.ui_mat = self.ui_mat.to_dense()

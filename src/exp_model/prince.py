"""
Based on the original implementation of paper "PRINCE: Provider-side Interpretability with Counterfactual Explanations in Recommender Systems"
Original repository: https://github.com/azinmatin/prince
"""

import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .base_model import UserVectorExpBaseModel

class PRINCE(UserVectorExpBaseModel):
    def __init__(self, rec_model, device, args, config=None):
        super(PRINCE, self).__init__(rec_model, device, args, config)
        self.n_users = rec_model.n_users
        self.n_items = rec_model.n_items
        self.alpha = config.get('alpha_prince', 0.15)
        self.epsilon = config.get('epsilon', 0.00008)
        self.adj_csr = None
        self.adj_csc = None
        self.mode = 'explicit'
        self.to(device)
        self.build_graph(rec_model.data_handler.train_df)

    def build_graph(self, train_df):
        
        train_df = train_df.drop_duplicates(subset=['user_id', 'item_id']).copy()
        
        user_ids = train_df['user_id'].values
        item_ids = train_df['item_id'].values
        
        n_users = self.n_users
        n_items = self.n_items
        n_nodes = n_users + n_items
        
        user_counts = train_df['user_id'].value_counts()
        item_counts = train_df['item_id'].value_counts()
        
        u_deg = np.zeros(n_users)
        u_deg[user_counts.index] = user_counts.values
        u_norm = 1.0 / (u_deg + 1)
        
        i_deg = np.zeros(n_items)
        i_deg[item_counts.index] = item_counts.values
        i_norm = 1.0 / (i_deg + 1)
        
        u_indices = user_ids
        i_indices = item_ids + n_users
        ui_weights = u_norm[user_ids]
        
        iu_weights = i_norm[item_ids]
        
        u_self = np.arange(n_users)
        uu_weights = u_norm
        
        i_self = np.arange(n_items) + n_users
        ii_weights = i_norm
        
        rows = np.concatenate([u_indices, i_indices, u_self, i_self])
        cols = np.concatenate([i_indices, u_indices, u_self, i_self])
        data = np.concatenate([ui_weights, iu_weights, uu_weights, ii_weights])
        
        self.adj_csr = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
        self.adj_csc = self.adj_csr.tocsc()

    def _reverse_local_push(self, target_node, alpha, epsilon, adj_csc, p=None, r=None, update=False):
        n_nodes = adj_csc.shape[0]
        if not update:
            p = np.zeros(n_nodes, dtype=np.float32)
            r = np.zeros(n_nodes, dtype=np.float32)
            r[target_node] = 1.0
            
        queue = [n for n in np.where(np.abs(r) > epsilon)[0]]
        in_queue = np.zeros(n_nodes, dtype=bool)
        in_queue[queue] = True
        
        idx = 0
        while idx < len(queue):
            u = queue[idx]
            idx += 1
            in_queue[u] = False
            
            res = r[u]
            if abs(res) <= epsilon:
                continue
                
            p[u] += alpha * res
            
            start = adj_csc.indptr[u]
            end = adj_csc.indptr[u+1]
            preds = adj_csc.indices[start:end]
            weights = adj_csc.data[start:end]
            
            r[preds] += (1 - alpha) * res * weights
            
            self_loop_mask = (preds == u)
            if np.any(self_loop_mask):
                self_weight = weights[self_loop_mask][0]
                r[u] = (1 - alpha) * self_weight * res
            else:
                r[u] = 0
            
            active_mask = (np.abs(r[preds]) > epsilon) & (~in_queue[preds])
            new_active = preds[active_mask]
            
            if len(new_active) > 0:
                queue.extend(new_active)
                in_queue[new_active] = True
                
        return p, r

    def _compute_delta_r(self, p, r, user_node, deleted_neighbors, deleted_weights, sum_deleted_weights, alpha):
        avg_deleted = np.dot(p[deleted_neighbors], deleted_weights) / sum_deleted_weights
        old_avg = (p[user_node] + alpha * r[user_node]) / (1 - alpha)
        tmp = (avg_deleted - old_avg) * sum_deleted_weights / (1.0 - sum_deleted_weights)
        return (tmp * (1 - alpha)) / alpha

    def _create_modified_adj(self, user_node):
        adj_mod = self.adj_csr.copy()
        
        start = adj_mod.indptr[user_node]
        end = adj_mod.indptr[user_node + 1]
        row_indices = adj_mod.indices[start:end]
        row_data = adj_mod.data[start:end]
        
        mask_del = (row_indices != user_node)
        deleted_neighbors = row_indices[mask_del].copy()
        deleted_weights = row_data[mask_del].copy()
        sum_deleted_weights = np.sum(deleted_weights)
        has_self_loop = np.any(~mask_del)
        
        row_data[mask_del] = 0.0
        
        if sum_deleted_weights < 1.0 and has_self_loop:
            scale = 1.0 / (1.0 - sum_deleted_weights)
            self_loop_pos = np.where(~mask_del)[0]
            for pos in self_loop_pos:
                row_data[pos] *= scale
        
        adj_mod.eliminate_zeros()
        adj_csc_mod = adj_mod.tocsc()
        
        return adj_csc_mod, deleted_neighbors, deleted_weights, sum_deleted_weights, has_self_loop

    def _compute_ppr_without_user(self, user_node, items, p_org, r_org, alpha, epsilon):
        adj_csc_mod, deleted_neighbors, deleted_weights, sum_del, has_self_loop = self._create_modified_adj(user_node)
        
        if len(deleted_neighbors) == 0:
            return {item: p_org[item].copy() for item in items}
        
        p_no_u = {}
        
        if has_self_loop and sum_del < 1.0:
            for item in items:
                p_curr = p_org[item].copy()
                r_curr = r_org[item].copy()
                
                delta_r = self._compute_delta_r(p_curr, r_curr, user_node, deleted_neighbors, deleted_weights, sum_del, alpha)
                r_curr[user_node] -= delta_r
                
                p_new, _ = self._reverse_local_push(item, alpha, epsilon, adj_csc_mod, p=p_curr, r=r_curr, update=True)
                p_no_u[item] = p_new
        else:
            for item in items:
                p_new, _ = self._reverse_local_push(item, alpha, epsilon, adj_csc_mod)
                p_no_u[item] = p_new
        
        return p_no_u

    def _run_ppr_job(self, node):
        p, r = self._reverse_local_push(node, self.alpha, self.epsilon, self.adj_csc)
        return node, p, r

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        if isinstance(item_ids, list) and len(item_ids) > 1:
            raise ValueError("PRINCE does not support collective list-level explicit explanation. Please set level='item'.")
            
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
            
        if not isinstance(item_ids, list):
            item_ids = [item_ids]
            
        top_k = self.args.top_k
        target_item = int(item_ids[0])
        
        history_mask = user_interaction > 0
        history_indices = torch.where(history_mask)[0].cpu().numpy()
        
        if len(history_indices) == 0:
            return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))
            
        target_node = self.n_users + target_item
        user_node = user_id 
        
        candidate_pool = []
        with torch.no_grad():
            user_mask = user_interaction.unsqueeze(0)
            user_tensor = torch.tensor([user_id], device=self.device, dtype=torch.long)
            scores = self.rec_model.predict(users=user_tensor, mask=[user_mask[0]])
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            _, topk_indices = torch.topk(scores[0], min(top_k + 1, scores.shape[1]))
            topk_items = topk_indices.cpu().tolist()
            for item in topk_items:
                if item != target_item:
                    candidate_pool.append(self.n_users + item)
                    
        if not candidate_pool:
             return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))
             
        nodes_to_compute = list(set([target_node] + candidate_pool))
        
        p_org = {}
        r_org = {}
        for n in nodes_to_compute:
            node, p, r = self._run_ppr_job(n)
            p_org[n] = p
            r_org[n] = r
        
        p_no_u = self._compute_ppr_without_user(
            user_node, nodes_to_compute, p_org, r_org, self.alpha, self.epsilon
        )
        
        start = self.adj_csr.indptr[user_node]
        end = self.adj_csr.indptr[user_node+1]
        neighbor_nodes = self.adj_csr.indices[start:end]
        neighbor_weights = self.adj_csr.data[start:end]
        
        mask = (neighbor_nodes != user_node)
        neighbors = neighbor_nodes[mask]
        weights = neighbor_weights[mask]
        
        if len(neighbors) == 0:
            return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))
            
        p_top_vals = p_no_u[target_node][neighbors]
        
        n_candidates = len(candidate_pool)
        n_neighbors = len(neighbors)
        
        p_cand_matrix = np.zeros((n_candidates, n_neighbors))
        
        for i, cand_node in enumerate(candidate_pool):
            p_cand_matrix[i] = p_no_u[cand_node][neighbors]
            
        diff_matrix = (p_cand_matrix - p_top_vals) * weights
        
        best_cand_idx = -1
        min_number = np.iinfo(np.int32).max
        max_diff = -float('inf')
        best_cfe = []
        
        for c_idx in range(n_candidates):
            diff_vals = diff_matrix[c_idx]
            sorted_indices = np.argsort(diff_vals)
            
            current_sum_diff = np.sum(diff_vals)
            current_sum_weight = np.sum(weights)
            
            deleted = []
            replaced = False
            
            for k in range(n_neighbors - 1):
                if k + 1 > min_number:
                    break
                    
                idx = sorted_indices[k]
                val = diff_vals[idx]
                w = weights[idx]
                
                current_sum_diff -= val
                current_sum_weight -= w
                deleted.append(neighbors[idx])
                
                if current_sum_diff > 2 * self.epsilon * current_sum_weight:
                    replaced = True
                    break
            
            if replaced:
                if len(deleted) < min_number:
                    min_number = len(deleted)
                    best_cfe = deleted
                    best_cand_idx = c_idx
                    max_diff = current_sum_diff
                elif len(deleted) == min_number:
                    if current_sum_diff > max_diff:
                        max_diff = current_sum_diff
                        best_cfe = deleted
                        best_cand_idx = c_idx
                        
        if best_cand_idx == -1:
            return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))
            
        cfe_item_ids = []
        for node in best_cfe:
            if node >= self.n_users:
                i_id = node - self.n_users
                cfe_item_ids.append(i_id)
                
        return self.convert_cf_list_to_mask(torch.tensor(cfe_item_ids, device=self.device, dtype=torch.long))

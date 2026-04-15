"""
Based on the original implementation of paper "PRINCE: Provider-side Interpretability with Counterfactual Explanations in Recommender Systems"
Original repository: https://github.com/azinmatin/prince
"""

import torch
import numpy as np
import scipy.sparse as sp
from .base_model import UserVectorExpBaseModel

class PRINCE(UserVectorExpBaseModel):
    def __init__(self, rec_model, device, args, config=None):
        super(PRINCE, self).__init__(rec_model, device, args, config)
        self.n_users = rec_model.n_users
        self.n_items = rec_model.n_items
        self.n_nodes = self.n_users + self.n_items
        self.alpha = config.get('alpha_prince', 0.15)
        self.epsilon = config.get('epsilon', 0.00008)
        self.adj_csr = None
        self.adj_csc = None
        self.mode = 'explicit'
        self._cache_user_id = None
        self._cache_p_no_u = None
        self._cache_neighbors = None
        self._cache_weights = None
        self._cache_candidate_pool = None
        
        self._csc_indptr = None
        self._csc_indices = None
        self._csc_data = None
        self._empty_skip_mask = None

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
        
        self._csc_indptr = self.adj_csc.indptr.tolist()
        self._csc_indices = self.adj_csc.indices.tolist()
        self._csc_data = self.adj_csc.data.tolist()
        self._empty_skip_mask = [False] * n_nodes

    def _reverse_local_push(self, target_node, alpha, epsilon, p=None, r=None, update=False, skip_node=-1, skip_mask=None, self_loop_scale=1.0):
        n_nodes = self.n_nodes
        if not update or p is None or r is None:
            p = [0.0] * n_nodes
            r = [0.0] * n_nodes
            r[target_node] = 1.0

        if skip_mask is None:
            skip_mask = self._empty_skip_mask

        indptr = self._csc_indptr
        indices = self._csc_indices
        data = self._csc_data

        in_queue = [False] * n_nodes
        out_r = []
        for i, val in enumerate(r):
            if abs(val) > epsilon:
                out_r.append(i)
                in_queue[i] = True

        one_minus_alpha = 1.0 - alpha

        while out_r:
            push_node = out_r.pop()
            in_queue[push_node] = False

            res = r[push_node]
            if abs(res) <= epsilon:
                continue

            p[push_node] += alpha * res

            start = indptr[push_node]
            end = indptr[push_node + 1]

            self_weight = 0.0
            push_val = one_minus_alpha * res

            if skip_mask[push_node]:
                for idx in range(start, end):
                    pred = indices[idx]
                    w = data[idx]

                    if pred == push_node:
                        self_weight = w
                        continue
                    
                    if pred == skip_node:
                        continue

                    r[pred] += push_val * w
                    if abs(r[pred]) > epsilon and not in_queue[pred]:
                        out_r.append(pred)
                        in_queue[pred] = True

                r[push_node] = push_val * self_weight
                if abs(r[push_node]) > epsilon and not in_queue[push_node]:
                    out_r.append(push_node)
                    in_queue[push_node] = True

            elif push_node == skip_node:
                for idx in range(start, end):
                    pred = indices[idx]
                    w = data[idx]

                    if pred == push_node:
                        self_weight = w * self_loop_scale
                        continue

                    r[pred] += push_val * w
                    if abs(r[pred]) > epsilon and not in_queue[pred]:
                        out_r.append(pred)
                        in_queue[pred] = True

                r[push_node] = push_val * self_weight
                if abs(r[push_node]) > epsilon and not in_queue[push_node]:
                    out_r.append(push_node)
                    in_queue[push_node] = True

            else:
                for idx in range(start, end):
                    pred = indices[idx]
                    w = data[idx]

                    if pred == push_node:
                        self_weight = w
                        continue

                    r[pred] += push_val * w
                    if abs(r[pred]) > epsilon and not in_queue[pred]:
                        out_r.append(pred)
                        in_queue[pred] = True

                r[push_node] = push_val * self_weight
                if abs(r[push_node]) > epsilon and not in_queue[push_node]:
                    out_r.append(push_node)
                    in_queue[push_node] = True

        return p, r

    def _get_user_edge_info(self, user_node):
        start = self.adj_csr.indptr[user_node]
        end = self.adj_csr.indptr[user_node + 1]
        row_indices = self.adj_csr.indices[start:end]
        row_data = self.adj_csr.data[start:end]

        mask_del = (row_indices != user_node)
        deleted_neighbors = row_indices[mask_del].copy()
        deleted_weights = row_data[mask_del].copy()
        sum_deleted_weights = np.sum(deleted_weights)
        has_self_loop = np.any(~mask_del)

        self_loop_scale = 1.0
        if sum_deleted_weights < 1.0 and has_self_loop:
            self_loop_scale = 1.0 / (1.0 - sum_deleted_weights)

        skip_mask = [False] * self.n_nodes
        for i in deleted_neighbors:
            skip_mask[int(i)] = True

        return deleted_neighbors.tolist(), deleted_weights.tolist(), sum_deleted_weights, has_self_loop, skip_mask, self_loop_scale

    def _compute_delta_r(self, p, r, user_node, deleted_neighbors, deleted_weights, sum_deleted_weights, alpha):
        avg_deleted = sum(p[n] * w for n, w in zip(deleted_neighbors, deleted_weights)) / sum_deleted_weights
        old_avg = (p[user_node] + alpha * r[user_node]) / (1 - alpha)
        tmp = (avg_deleted - old_avg) * sum_deleted_weights / (1.0 - sum_deleted_weights)
        return (tmp * (1 - alpha)) / alpha

    def _compute_ppr_without_user(self, user_node, items, p_org, r_org, alpha, epsilon):
        deleted_neighbors, deleted_weights, sum_del, has_self_loop, skip_mask, self_loop_scale = self._get_user_edge_info(user_node)

        if len(deleted_neighbors) == 0:
            return {item: list(p_org[item]) for item in items}

        p_no_u = {}

        if has_self_loop and sum_del < 1.0:
            for item in items:
                p_curr = list(p_org[item])
                r_curr = list(r_org[item])

                delta_r = self._compute_delta_r(p_curr, r_curr, user_node, deleted_neighbors, deleted_weights, sum_del, alpha)
                r_curr[user_node] -= delta_r

                p_new, _ = self._reverse_local_push(
                    item, alpha, epsilon, 
                    p=p_curr, r=r_curr, update=True, 
                    skip_node=user_node, skip_mask=skip_mask, self_loop_scale=self_loop_scale
                )
                p_no_u[item] = p_new
        else:
            for item in items:
                p_new, _ = self._reverse_local_push(
                    item, alpha, epsilon, 
                    skip_node=user_node, skip_mask=skip_mask, self_loop_scale=self_loop_scale
                )
                p_no_u[item] = p_new

        return p_no_u

    def _run_ppr_job(self, node):
        p, r = self._reverse_local_push(node, self.alpha, self.epsilon)
        return node, p, r

    def _ensure_user_cache(self, user_id):
        if self._cache_user_id == user_id:
            return

        top_k = self.args.top_k
        user_node = user_id

        with torch.no_grad():
            user_tensor = torch.tensor([user_id], device=self.device, dtype=torch.long)
            scores = self.rec_model.predict(users=user_tensor)
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)
            _, topk_indices = torch.topk(scores[0], min(top_k + 1, scores.shape[1]))
            topk_items = topk_indices.cpu().tolist()

        all_item_nodes = list(set([self.n_users + item for item in topk_items]))

        p_org = {}
        r_org = {}
        for n in all_item_nodes:
            _, p, r = self._run_ppr_job(n)
            p_org[n] = p
            r_org[n] = r

        p_no_u = self._compute_ppr_without_user(
            user_node, all_item_nodes, p_org, r_org, self.alpha, self.epsilon
        )

        start = self.adj_csr.indptr[user_node]
        end = self.adj_csr.indptr[user_node + 1]
        neighbor_nodes = self.adj_csr.indices[start:end]
        neighbor_weights = self.adj_csr.data[start:end]
        mask = (neighbor_nodes != user_node)
        neighbors = neighbor_nodes[mask].tolist()
        weights = neighbor_weights[mask].tolist()

        self._cache_user_id = user_id
        self._cache_p_no_u = p_no_u
        self._cache_neighbors = neighbors
        self._cache_weights = weights
        self._cache_candidate_pool = all_item_nodes

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        if isinstance(item_ids, list) and len(item_ids) > 1:
            raise ValueError("PRINCE does not support collective list-level explicit explanation. Please set level='item'.")

        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)

        if not isinstance(item_ids, list):
            item_ids = [item_ids]

        target_item = int(item_ids[0])

        history_mask = user_interaction > 0
        history_indices = torch.where(history_mask)[0].cpu().numpy()

        if len(history_indices) == 0:
            return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))

        target_node = self.n_users + target_item

        self._ensure_user_cache(user_id)

        p_no_u = self._cache_p_no_u
        neighbors = self._cache_neighbors
        weights = self._cache_weights

        candidate_pool = [n for n in self._cache_candidate_pool if n != target_node]

        if len(neighbors) == 0 or len(candidate_pool) == 0:
            return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))

        if target_node not in p_no_u:
            return self.convert_cf_list_to_mask(torch.tensor([], device=self.device, dtype=torch.long))

        # Vectorized lookup for speed
        p_target = p_no_u[target_node]
        p_top_vals = np.array([p_target[n] for n in neighbors])

        n_candidates = len(candidate_pool)
        n_neighbors = len(neighbors)

        p_cand_matrix = np.zeros((n_candidates, n_neighbors))
        for i, cand_node in enumerate(candidate_pool):
            p_cand = p_no_u[cand_node]
            p_cand_matrix[i] = [p_cand[n] for n in neighbors]

        weights_arr = np.array(weights)
        diff_matrix = (p_cand_matrix - p_top_vals) * weights_arr

        best_cand_idx = -1
        min_number = np.iinfo(np.int32).max
        max_diff = -float('inf')
        best_cfe = []

        for c_idx in range(n_candidates):
            diff_vals = diff_matrix[c_idx]
            sorted_indices = np.argsort(diff_vals)

            current_sum_diff = np.sum(diff_vals)
            current_sum_weight = np.sum(weights_arr)

            deleted = []
            replaced = False

            for k in range(n_neighbors - 1):
                if k + 1 > min_number:
                    break

                idx = sorted_indices[k]
                val = diff_vals[idx]
                w = weights_arr[idx]

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
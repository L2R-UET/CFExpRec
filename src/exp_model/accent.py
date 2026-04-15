'''
Based on the original implementation of paper "Counterfactual Explanations for Neural Recommenders"
Original repository: https://github.com/hieptk/accent
'''

import torch
import torch.nn.functional as F
import numpy as np
from .base_model import UserVectorExpBaseModel

class ACCENT(UserVectorExpBaseModel):

    def __init__(self, rec_model, device, args, config=None):
        super(ACCENT, self).__init__(rec_model, device, args, config)
        self.n_users = rec_model.n_users
        self.n_items = rec_model.n_items
        self.mode = 'hybrid'
        self.to(device)

        if self._is_vae() or self._is_diffrec():
            return

        self.damping = config.get('damping', 1e-6)
        self.cg_iters = config.get('cg_iters', 100)
        self.hvp_batch_size = config.get('hvp_batch_size', 4096)
        self.full_train_data = None
        self.num_train_examples = 0
        self.item_to_train_indices = {}
        self.train_users_t = None
        self.train_pos_t = None
        self.train_neg_t = None
        self._dense_ui_mat = rec_model.ui_mat.to_dense()
        self._init_train_data(rec_model.data_handler.train_df, n_items=self.n_items)

    def _is_mf(self):
        return self.rec_model.__class__.__name__ == 'MF'

    def _is_vae(self):
        return self.rec_model.__class__.__name__ == 'VAE'

    def _is_diffrec(self):
        return self.rec_model.__class__.__name__ == 'DiffRec'

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        if self._is_vae() or self._is_diffrec():
            return self._perturb_implicit(user_id, item_ids)
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
        user_interactions_t = user_interaction.unsqueeze(0)

        history_mask = user_interaction > 0
        history_indices = torch.where(history_mask)[0]
        history_indices_np = history_indices.cpu().numpy()

        target_items = item_ids if isinstance(item_ids, (list, np.ndarray)) else [item_ids]

        if len(history_indices_np) == 0:
             return torch.zeros(self.n_items, device=self.device)

        influences_matrix, history_indices_out = self._influence_topk(
            user_id, target_items, user_interactions_t, history_indices_np
        )
        
        n_targets = influences_matrix.shape[0] if influences_matrix is not None else len(target_items)
        scores_matrix = torch.zeros((n_targets, self.n_items), device=self.device)

        if influences_matrix is not None and influences_matrix.size > 0 and len(history_indices_out) > 0:
            indices = [int(h) for h in history_indices_out]
            for i in range(n_targets):
                values = torch.tensor(influences_matrix[i], device=self.device, dtype=torch.float32)
                scores_matrix[i, indices] = values

        scores_tensor = scores_matrix.mean(dim=0)
        
        mask = torch.zeros(self.n_items, device=self.device)
        mask[history_indices] = 1.0
        scores_tensor = scores_tensor * mask
        scores_tensor += user_interaction * 1e-12
        return scores_tensor

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        if self._is_vae() or self._is_diffrec():
            return self._perturb_explicit(user_id, item_ids)
        if isinstance(item_ids, list) and len(item_ids) > 1:
            return torch.zeros(self.n_items, device=self.device)
            
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
        user_interactions_t = user_interaction.unsqueeze(0)

        top_k = self.args.top_k
        if not isinstance(item_ids, list):
            item_ids = [item_ids]

        history_mask = user_interactions_t[0] > 0
        history_indices = torch.where(history_mask)[0].cpu().numpy()
        n_history = len(history_indices)

        if n_history == 0:
            return torch.zeros(self.n_items, device=self.device)

        with torch.no_grad():
            user_tensor = torch.tensor([user_id], device=self.device, dtype=torch.long)
            base_scores = self.rec_model.predict(users=user_tensor)
            scores_np = base_scores[0].cpu().numpy()
            filtered_scores = base_scores + self.rec_model._history_mask[user_tensor]
            _, topk_indices = torch.topk(filtered_scores[0], top_k)
            topk_items = topk_indices.cpu().numpy()

        all_targets = list(set([int(item) for item in item_ids]) | set(topk_items.tolist()))
        influences, _ = self._influence_topk(user_id, all_targets, user_interactions_t, history_indices)
        
        item_to_inf_idx = {item: idx for idx, item in enumerate(all_targets)}

        cf_results = []
        for target_item in item_ids:
            target_int = int(target_item)
            if target_int not in item_to_inf_idx:
                continue
                
            best_res = None
            best_gap = float('inf')
            target_inf_idx = item_to_inf_idx[target_int]
            score_target = scores_np[target_int]

            for repl_item in topk_items:
                repl_int = int(repl_item)
                if repl_int == target_int:
                    continue
                    
                repl_inf_idx = item_to_inf_idx[repl_int]
                
                score_gap = score_target - scores_np[repl_int]
                gap_influences = influences[target_inf_idx] - influences[repl_inf_idx]
                
                tmp_res, tmp_gap = self._try_swap(score_gap, gap_influences)
                if tmp_res is not None:
                    if best_res is None or len(tmp_res) < len(best_res) or (len(tmp_res) == len(best_res) and tmp_gap < best_gap):
                        best_res = tmp_res
                        best_gap = tmp_gap

            if best_res is not None:
                cfe_items = [int(history_indices[idx]) for idx in best_res]
                cf_results.extend(cfe_items)

        return self.convert_cf_list_to_mask(torch.tensor(cf_results, device=self.device, dtype=torch.long))

    def _perturb_scores(self, user_id, history_indices):
        n_history = len(history_indices)
        user_t = torch.tensor([user_id], device=self.device, dtype=torch.long)
        masks = torch.ones((n_history, self.n_items), device=self.device)
        masks[torch.arange(n_history, device=self.device), history_indices] = 0
        users_batch = user_t.expand(n_history)
        with torch.no_grad():
            return self.rec_model.predict(users=users_batch, mask=masks)

    def _perturb_implicit(self, user_id, item_ids):
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
        history_mask = user_interaction > 0
        history_indices = torch.where(history_mask)[0]
        target_items = item_ids if isinstance(item_ids, (list, np.ndarray)) else [item_ids]
        if len(history_indices) == 0:
            return torch.zeros(self.n_items, device=self.device)
        user_t = torch.tensor([user_id], device=self.device, dtype=torch.long)
        with torch.no_grad():
            base_scores = self.rec_model.predict(users=user_t)
        perturb_scores = self._perturb_scores(user_id, history_indices)
        target_tensor = torch.tensor(target_items, device=self.device, dtype=torch.long)
        base_target = base_scores[0, target_tensor]
        perturb_target = perturb_scores[:, target_tensor]
        score_drops = (base_target.unsqueeze(0) - perturb_target).mean(dim=1)
        importance = torch.zeros(self.n_items, device=self.device)
        importance[history_indices] = score_drops
        importance += user_interaction * 1e-12
        return importance

    def _perturb_explicit(self, user_id, item_ids):
        if isinstance(item_ids, list) and len(item_ids) > 1:
            return torch.zeros(self.n_items, device=self.device)
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
        top_k = self.args.top_k
        if not isinstance(item_ids, list):
            item_ids = [item_ids]
        history_mask = user_interaction > 0
        history_indices = torch.where(history_mask)[0]
        n_history = len(history_indices)
        if n_history == 0:
            return torch.zeros(self.n_items, device=self.device)
        user_t = torch.tensor([user_id], device=self.device, dtype=torch.long)
        with torch.no_grad():
            base_scores = self.rec_model.predict(users=user_t)
            scores_np = base_scores[0].cpu().numpy()
            filtered_scores = base_scores + self.rec_model._history_mask[user_t]
            _, topk_indices = torch.topk(filtered_scores[0], top_k)
            topk_items = topk_indices.cpu().numpy()
        all_targets = list(set([int(item) for item in item_ids]) | set(topk_items.tolist()))
        target_tensor = torch.tensor(all_targets, device=self.device, dtype=torch.long)
        perturb_scores = self._perturb_scores(user_id, history_indices)
        base_target = base_scores[0, target_tensor].cpu().numpy()
        perturb_target = perturb_scores[:, target_tensor].cpu().numpy()
        influences = (base_target[np.newaxis, :] - perturb_target).T
        item_to_inf_idx = {item: idx for idx, item in enumerate(all_targets)}
        cf_results = []
        for target_item in item_ids:
            target_int = int(target_item)
            if target_int not in item_to_inf_idx:
                continue
            best_res = None
            best_gap = float('inf')
            target_inf_idx = item_to_inf_idx[target_int]
            score_target = scores_np[target_int]
            for repl_item in topk_items:
                repl_int = int(repl_item)
                if repl_int == target_int:
                    continue
                repl_inf_idx = item_to_inf_idx[repl_int]
                score_gap = score_target - scores_np[repl_int]
                gap_influences = influences[target_inf_idx] - influences[repl_inf_idx]
                tmp_res, tmp_gap = self._try_swap(score_gap, gap_influences)
                if tmp_res is not None:
                    if best_res is None or len(tmp_res) < len(best_res) or (len(tmp_res) == len(best_res) and tmp_gap < best_gap):
                        best_res = tmp_res
                        best_gap = tmp_gap
            if best_res is not None:
                cfe_items = [int(history_indices[idx]) for idx in best_res]
                cf_results.extend(cfe_items)
        return self.convert_cf_list_to_mask(torch.tensor(cf_results, device=self.device, dtype=torch.long))

    def _init_train_data(self, train_data, n_items=None):
        if n_items is not None and hasattr(train_data, 'itertuples'):
             users = train_data['user_id'].values
             items = train_data['item_id'].values
             neg_items = (users + items + 1337) % n_items
             self.full_train_data = np.column_stack((users, items, neg_items))
        elif not isinstance(train_data, np.ndarray):
             self.full_train_data = np.array(train_data)
        else:
             self.full_train_data = train_data
        self.num_train_examples = len(self.full_train_data)
        self.train_users_t = torch.tensor(self.full_train_data[:, 0], device=self.device, dtype=torch.long)
        self.train_pos_t = torch.tensor(self.full_train_data[:, 1], device=self.device, dtype=torch.long)
        self.train_neg_t = torch.tensor(self.full_train_data[:, 2], device=self.device, dtype=torch.long)
        self.train_map = {}
        self.user_to_train_indices = {}
        self.item_to_train_indices = {}
        for idx, row in enumerate(self.full_train_data):
            u, i = int(row[0]), int(row[1])
            self.train_map[(u, i)] = idx
            if u not in self.user_to_train_indices:
                self.user_to_train_indices[u] = []
            self.user_to_train_indices[u].append(idx)
            if i not in self.item_to_train_indices:
                self.item_to_train_indices[i] = []
            self.item_to_train_indices[i].append(idx)

    def _influence_topk(self, user_id, topk_items, user_interactions, history_items):
        if user_interactions.dim() == 1:
            user_interactions = user_interactions.unsqueeze(0)

        history_indices = history_items
        n_history = len(history_indices)
        if n_history == 0:
            return np.zeros((len(topk_items), 0)), history_indices

        self.rec_model.train()
        for p in self.rec_model.parameters():
            p.requires_grad = True

        params = self._get_params()
        influences = np.zeros((len(topk_items), n_history))

        multi_train_grads = self._train_grads(
            user_id, history_indices, params, [int(t) for t in topk_items])

        for k_idx, target_item in enumerate(topk_items):
            target_item_int = int(target_item)
            hvp_train_indices = self._related_indices(user_id, target_item_int)
            scale = len(hvp_train_indices) if len(hvp_train_indices) > 0 else self.num_train_examples

            test_loss = self._test_loss(user_id, target_item_int, user_interactions)
            test_grad_all = torch.autograd.grad(test_loss, params, create_graph=False, retain_graph=False, allow_unused=True)
            test_grad_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(test_grad_all, params)]
            test_grad = self._slice_grads(test_grad_all, params, target_item_int)
            batches = self._make_batches(hvp_train_indices)
            hvp_fn = lambda v, ti=target_item_int, b=batches: self._hvp_sliced(params, v, ti, precomputed_batches=b)
            inverse_hvp = self._inv_hvp(test_grad, hvp_fn)

            train_grads = multi_train_grads[target_item_int]
            for i, h in enumerate(history_indices):
                grad_h = train_grads[int(h)]
                influences[k_idx, i] = torch.dot(grad_h, inverse_hvp).item() / max(scale, 1)

        self.rec_model.eval()
        for p in self.rec_model.parameters():
            p.requires_grad = False

        return influences, history_indices

    def _test_loss(self, user_id, target_item, user_interactions):
        target_item_t = torch.tensor([target_item], device=self.device, dtype=torch.long)
        if user_interactions.dim() == 1:
            user_interactions = user_interactions.unsqueeze(0)
        user_vec = self.rec_model.users_fc(user_interactions)
        item_vec = self.rec_model.item_fc(target_item_t)
        if hasattr(self.rec_model, 'item_bias'):
            item_vec = item_vec + self.rec_model.item_bias
        score = (user_vec * item_vec).sum(dim=1)
        return -score[0]

    def _sample_loss(self, user_id, history_indices):
        user_ids_batch = []
        pos_items_batch = []
        neg_items_batch = []

        for hist_item in history_indices:
            key = (int(user_id), int(hist_item))
            if key in self.train_map:
                row_idx = self.train_map[key]
                row = self.full_train_data[row_idx]
                neg_item = int(row[2]) if len(row) > 2 else (int(user_id) + int(hist_item) + 1337) % self.n_items
            else:
                neg_item = (int(user_id) + int(hist_item) + 1337) % self.n_items
            user_ids_batch.append(user_id)
            pos_items_batch.append(int(hist_item))
            neg_items_batch.append(neg_item)

        user_ids_t = torch.tensor(user_ids_batch, device=self.device, dtype=torch.long)
        pos_items_t = torch.tensor(pos_items_batch, device=self.device, dtype=torch.long)
        neg_items_t = torch.tensor(neg_items_batch, device=self.device, dtype=torch.long)

        user_histories = self._dense_ui_mat[user_ids_t]
        user_vecs = self.rec_model.users_fc(user_histories)
        pos_vecs = self.rec_model.item_fc(pos_items_t)
        if hasattr(self.rec_model, 'item_bias'):
            pos_vecs = pos_vecs + self.rec_model.item_bias
        pos_scores = torch.sigmoid((user_vecs * pos_vecs).sum(dim=1))
        neg_vecs = self.rec_model.item_fc(neg_items_t)
        if hasattr(self.rec_model, 'item_bias'):
            neg_vecs = neg_vecs + self.rec_model.item_bias
        neg_scores = torch.sigmoid((user_vecs * neg_vecs).sum(dim=1))
        per_sample_loss = -torch.log(pos_scores + 1e-10) - torch.log(1 - neg_scores + 1e-10)
        
        return per_sample_loss

    def _train_grads(self, user_id, history_indices, params, target_items):
        result = {int(t): {} for t in target_items}
        n_history = len(history_indices)
        if n_history == 0:
            return result

        per_sample_loss = self._sample_loss(user_id, history_indices)
        weight_matrix = torch.eye(n_history, device=self.device)
        for idx, hist_item in enumerate(history_indices):
            weighted_loss = (per_sample_loss * weight_matrix[idx]).sum()
            grads = torch.autograd.grad(weighted_loss, params, retain_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
            for target_item in target_items:
                result[int(target_item)][int(hist_item)] = self._slice_grads(grads, params, int(target_item)).detach()

        return result

    def _batch_train_loss(self, user_ids, pos_items, neg_items):
        user_histories = self._dense_ui_mat[user_ids]
        user_vecs = self.rec_model.users_fc(user_histories)
        pos_vecs = self.rec_model.item_fc(pos_items)
        if hasattr(self.rec_model, 'item_bias'):
            pos_vecs = pos_vecs + self.rec_model.item_bias
        pos_scores = torch.sigmoid((user_vecs * pos_vecs).sum(dim=1))
        neg_vecs = self.rec_model.item_fc(neg_items)
        if hasattr(self.rec_model, 'item_bias'):
            neg_vecs = neg_vecs + self.rec_model.item_bias
        neg_scores = torch.sigmoid((user_vecs * neg_vecs).sum(dim=1))
        return F.binary_cross_entropy(pos_scores, torch.ones_like(pos_scores)) + \
               F.binary_cross_entropy(neg_scores, torch.zeros_like(neg_scores))

    def _make_batches(self, train_indices):
        if train_indices is None or len(train_indices) == 0:
            active_indices = np.arange(len(self.full_train_data))
        else:
            active_indices = train_indices
        num_active = len(active_indices)
        if num_active == 0:
            return []
        batches = []
        for start_idx in range(0, num_active, self.hvp_batch_size):
            end_idx = min(start_idx + self.hvp_batch_size, num_active)
            idx = active_indices[start_idx:end_idx]
            batches.append((
                self.train_users_t[idx],
                self.train_pos_t[idx],
                self.train_neg_t[idx],
                end_idx - start_idx
            ))
        return batches

    def _hvp_sliced(self, params, v_sub, target_item, train_indices=None, precomputed_batches=None):
        if self.full_train_data is None or len(self.full_train_data) == 0:
            return self.damping * v_sub

        if precomputed_batches is not None:
            batches = precomputed_batches
        else:
            batches = self._make_batches(train_indices)

        if len(batches) == 0:
            return self.damping * v_sub

        hvp_total = torch.zeros_like(v_sub)
        total_samples = 0

        for user_ids_t, pos_items_t, neg_items_t, batch_size in batches:
            batch_loss = self._batch_train_loss(user_ids_t, pos_items_t, neg_items_t)

            grad_all = torch.autograd.grad(batch_loss, params, create_graph=True, retain_graph=True, allow_unused=True)
            grad_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_all, params)]

            grad_sub = self._slice_grads(grad_all, params, target_item)

            dot = torch.dot(grad_sub, v_sub.detach())

            grad2_all = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
            grad2_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad2_all, params)]
            hvp_sub = self._slice_grads(grad2_all, params, target_item)

            hvp_total += hvp_sub.detach() * batch_size
            total_samples += batch_size

        return hvp_total / float(total_samples) + self.damping * v_sub

    def _inv_hvp(self, v, hvp_fn):
        x = torch.zeros_like(v)
        r = v.detach().clone()
        p = r.clone()
        rsold = torch.dot(r, r)

        for _ in range(self.cg_iters):
            Ap = hvp_fn(p)
            alpha = rsold / (torch.dot(p, Ap) + 1e-10)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = torch.dot(r, r)
            if torch.sqrt(rsnew) < 1e-5:
                break
            p = r + (rsnew / (rsold + 1e-10)) * p
            rsold = rsnew
        return x

    def _related_indices(self, user_id, target_item):
        user_indices = self.user_to_train_indices.get(user_id, [])
        item_indices = self.item_to_train_indices.get(target_item, [])
        combined = set(user_indices) | set(item_indices)
        return np.array(sorted(combined)) if combined else np.array([], dtype=int)

    def _try_swap(self, score_gap, gap_influences):
        sorted_indices = np.argsort(-gap_influences)
        removed_items = set()
        current_gap = score_gap

        for idx in sorted_indices:
            if gap_influences[idx] < 0:
                break
            removed_items.add(idx)
            current_gap -= gap_influences[idx]
            if current_gap < 0:
                break

        if current_gap < 0:
            return removed_items, current_gap
        return None, float('inf')

    def _get_params(self):
        return [p for p in self.rec_model.parameters() if p.requires_grad]

    def _flatten_grads(self, grads):
        return torch.cat([g.reshape(-1) for g in grads])

    def _slice_grads(self, grads_list, params, target_item):
        sliced = []
        for g, p in zip(grads_list, params):
            if p is self.rec_model.item_fc.weight:
                sliced.append(g[target_item].reshape(-1))
            else:
                sliced.append(g.reshape(-1))
        return torch.cat(sliced)

    def _grad(self, loss, params):
        grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True, allow_unused=True)
        grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
        return self._flatten_grads(grads)
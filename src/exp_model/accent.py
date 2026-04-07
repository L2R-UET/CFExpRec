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
        self.damping = config.get('damping', 1e-6)
        self.cg_iters = config.get('cg_iters', 100)
        self.hvp_batch_size = config.get('hvp_batch_size', 4096)
        self.n_users = rec_model.n_users
        self.n_items = rec_model.n_items
        self.full_train_data = None
        self.num_train_examples = 0
        self.mode = 'hybrid'
        self.item_to_train_indices = {}
        self.to(device)
        self.prepare_data(rec_model.data_handler.train_df, n_items=self.n_items)

    def prepare_data(self, train_data, n_items=None):
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

    def _is_mlp(self):
        return hasattr(self.rec_model, 'users_fc') and hasattr(self.rec_model, 'item_fc')

    def _is_vae(self):
        return hasattr(self.rec_model, 'encode') and hasattr(self.rec_model, 'decode')

    def _is_diffrec(self):
        return hasattr(self.rec_model, 'diffusion') and hasattr(self.rec_model, 'model') and hasattr(self.rec_model, 'sampling_steps')

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
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
                scores_matrix[i, indices] = -values

        scores_tensor = scores_matrix.mean(dim=0)
        
        mask = torch.zeros(self.n_items, device=self.device)
        mask[history_indices] = 1.0
        scores_tensor = scores_tensor * mask
        scores_tensor += user_interaction * 1e-12
        
        return scores_tensor

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
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
            base_scores = self.rec_model.predict(users=user_tensor, mask=[user_interactions_t[0]])
            scores_np = base_scores[0].cpu().numpy()
            _, topk_indices = torch.topk(base_scores[0], top_k)
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
                gap_influences = influences[repl_inf_idx] - influences[target_inf_idx]
                
                tmp_res, tmp_gap = self._try_swap(score_gap, gap_influences)
                if tmp_res is not None:
                    if best_res is None or len(tmp_res) < len(best_res) or (len(tmp_res) == len(best_res) and tmp_gap < best_gap):
                        best_res = tmp_res
                        best_gap = tmp_gap

            if best_res is not None:
                cfe_items = [int(history_indices[idx]) for idx in best_res]
                cf_results.extend(cfe_items)

        return self.convert_cf_list_to_mask(torch.tensor(cf_results, device=self.device, dtype=torch.long))
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
        use_sliced = self._is_mlp()

        influences = np.zeros((len(topk_items), n_history))

        if use_sliced:
            multi_train_grads = self._iter_train_grads_sliced(
                user_id, history_indices, params, [int(t) for t in topk_items])

            for k_idx, target_item in enumerate(topk_items):
                target_item_int = int(target_item)
                hvp_train_indices = self._get_train_indices_of_test_case(user_id, target_item_int)
                scale = len(hvp_train_indices) if len(hvp_train_indices) > 0 else self.num_train_examples

                test_loss = self._test_loss(user_id, target_item_int, user_interactions)
                test_grad_all = torch.autograd.grad(test_loss, params, create_graph=False, retain_graph=False, allow_unused=True)
                test_grad_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(test_grad_all, params)]
                test_grad = self._slice_grads(test_grad_all, params, target_item_int)
                hvp_fn = lambda v, ti=target_item_int, hti=hvp_train_indices: self._hvp_sliced(params, v, ti, hti)
                inverse_hvp = self._inv_hvp(test_grad, hvp_fn)

                train_grads = multi_train_grads[target_item_int]
                for i, h in enumerate(history_indices):
                    grad_h = train_grads[int(h)]
                    influences[k_idx, i] = torch.dot(grad_h, inverse_hvp).item() / max(scale, 1)

        else:
            inv_hvps = []
            scales = []

            for k_idx, target_item in enumerate(topk_items):
                target_item_int = int(target_item)

                hvp_train_indices = self._get_train_indices_of_test_case(user_id, target_item_int)

                scale = len(hvp_train_indices) if len(hvp_train_indices) > 0 else self.num_train_examples
                scales.append(scale)

                test_loss = self._test_loss(user_id, target_item_int, user_interactions)
                test_grad_list = torch.autograd.grad(test_loss, params, create_graph=False, retain_graph=False, allow_unused=True)
                test_grad_list = [g if g is not None else torch.zeros_like(p) for g, p in zip(test_grad_list, params)]
                test_grad = self._flatten_grads(test_grad_list)
                hvp_fn = lambda v, hti=hvp_train_indices: self._hvp(params, v, hti)
                inverse_hvp = self._inv_hvp(test_grad, hvp_fn)
                inv_hvps.append(inverse_hvp.detach())

            h_to_idx = {int(h): i for i, h in enumerate(history_indices)}

            for h_val, grad_h in self._iter_train_grads(user_id, history_indices, params):
                h_idx = h_to_idx[h_val]
                for k_idx in range(len(topk_items)):
                    influences[k_idx, h_idx] = torch.dot(grad_h, inv_hvps[k_idx]).item() / max(scales[k_idx], 1)

        self.rec_model.eval()
        for p in self.rec_model.parameters():
            p.requires_grad = False

        return influences, history_indices

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

    def _compute_per_sample_loss(self, user_id, history_indices):
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

        if self._is_vae():
            user_interaction = self.rec_model.ui_mat.index_select(0, user_ids_t[:1]).to_dense()
            mu, logvar = self.rec_model.encode(user_interaction)
            z = mu
            logits = self.rec_model.decode(z)
            log_softmax_var = F.log_softmax(logits, dim=1)
            per_sample_loss = -log_softmax_var[0, pos_items_t]
        elif self._is_diffrec():
            user_interaction = self.rec_model.ui_mat.index_select(0, user_ids_t[:1]).to_dense()
            t = torch.zeros(1, device=self.device, dtype=torch.long)
            model_output = self.rec_model.model(user_interaction, t)
            per_sample_loss = (user_interaction[0, pos_items_t] - model_output[0, pos_items_t]) ** 2
        else:
            user_histories = self.rec_model.ui_mat.index_select(0, user_ids_t).to_dense()
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

    def _iter_train_grads(self, user_id, history_indices, params):
        n_history = len(history_indices)
        if n_history == 0:
            return

        per_sample_loss = self._compute_per_sample_loss(user_id, history_indices)
        weight_matrix = torch.eye(n_history, device=self.device)
        for idx, hist_item in enumerate(history_indices):
            weighted_loss = (per_sample_loss * weight_matrix[idx]).sum()
            grads = torch.autograd.grad(weighted_loss, params, retain_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
            yield int(hist_item), self._flatten_grads(grads).detach()

    def _iter_train_grads_sliced(self, user_id, history_indices, params, target_items):
        result = {int(t): {} for t in target_items}
        n_history = len(history_indices)
        if n_history == 0:
            return result

        per_sample_loss = self._compute_per_sample_loss(user_id, history_indices)
        weight_matrix = torch.eye(n_history, device=self.device)
        for idx, hist_item in enumerate(history_indices):
            weighted_loss = (per_sample_loss * weight_matrix[idx]).sum()
            grads = torch.autograd.grad(weighted_loss, params, retain_graph=True, allow_unused=True)
            grads = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, params)]
            for target_item in target_items:
                result[int(target_item)][int(hist_item)] = self._slice_grads(grads, params, int(target_item)).detach()

        return result

    def _test_loss(self, user_id, target_item, user_interactions):
        target_item_t = torch.tensor([target_item], device=self.device, dtype=torch.long)

        if self._is_vae():
            if user_interactions.dim() == 1:
                user_interactions = user_interactions.unsqueeze(0)
            mu, _ = self.rec_model.encode(user_interactions)
            logits = self.rec_model.decode(mu)
            score = logits[0, target_item].unsqueeze(0)
        elif self._is_diffrec():
            if user_interactions.dim() == 1:
                user_interactions = user_interactions.unsqueeze(0)
            t = torch.zeros(1, device=self.device, dtype=torch.long)
            model_output = self.rec_model.model(user_interactions, t)
            score = model_output[0, target_item].unsqueeze(0)
        else:
            if user_interactions.dim() == 1:
                user_interactions = user_interactions.unsqueeze(0)
            user_vec = self.rec_model.users_fc(user_interactions)
            item_vec = self.rec_model.item_fc(target_item_t)
            if hasattr(self.rec_model, 'item_bias'):
                item_vec = item_vec + self.rec_model.item_bias
            score = (user_vec * item_vec).sum(dim=1)
        return -score[0]

    def _batch_train_loss(self, user_ids, pos_items, neg_items):
        if not torch.is_tensor(user_ids):
             user_ids = torch.tensor(user_ids, device=self.device, dtype=torch.long)
        if not torch.is_tensor(pos_items):
             pos_items = torch.tensor(pos_items, device=self.device, dtype=torch.long)
        if not torch.is_tensor(neg_items):
             neg_items = torch.tensor(neg_items, device=self.device, dtype=torch.long)

        if self._is_vae():
            unique_users = torch.unique(user_ids)
            batch_ratings = self.rec_model.ui_mat.index_select(0, unique_users).to_dense()
            mu, logvar = self.rec_model.encode(batch_ratings)
            z = mu
            logits = self.rec_model.decode(z)
            log_softmax_var = F.log_softmax(logits, dim=1)
            neg_ll = -torch.sum(log_softmax_var * batch_ratings, dim=1)
            kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            anneal = getattr(self.rec_model, 'anneal_cap', 0.2)
            return torch.mean(neg_ll + anneal * kl_div)
        elif self._is_diffrec():
            unique_users = torch.unique(user_ids)
            batch_ratings = self.rec_model.ui_mat.index_select(0, unique_users).to_dense()
            t = torch.zeros(len(unique_users), device=self.device, dtype=torch.long)
            model_output = self.rec_model.model(batch_ratings, t)
            return F.mse_loss(model_output, batch_ratings)
        else:
            user_histories = self.rec_model.ui_mat.index_select(0, user_ids).to_dense()
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

    def _hvp(self, params, v, train_indices=None):
        if self.full_train_data is None or len(self.full_train_data) == 0:
            return self.damping * v

        active_indices = train_indices if train_indices is not None and len(train_indices) > 0 else np.arange(len(self.full_train_data))
        num_active = len(active_indices)
        if num_active == 0:
             return self.damping * v

        if self._is_vae() or self._is_diffrec():
            all_user_ids = np.array([int(self.full_train_data[i][0]) for i in active_indices])
            unique_user_ids = np.unique(all_user_ids)
            hvp_total = torch.zeros_like(v)
            total_interactions = num_active

            for start_idx in range(0, len(unique_user_ids), max(1, self.hvp_batch_size // 50)):
                end_idx = min(start_idx + max(1, self.hvp_batch_size // 50), len(unique_user_ids))
                batch_users = unique_user_ids[start_idx:end_idx]
                n_batch_users = len(batch_users)

                user_ids_t = torch.tensor(batch_users, device=self.device, dtype=torch.long)
                dummy = torch.zeros(n_batch_users, device=self.device, dtype=torch.long)

                batch_loss = self._batch_train_loss(user_ids_t, dummy, dummy)
                grad = self._grad(batch_loss, params)

                hvp_batch = torch.autograd.grad(grad, params, grad_outputs=v, retain_graph=True, allow_unused=True)
                hvp_batch = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_batch, params)]
                hvp_total += self._flatten_grads(hvp_batch) * n_batch_users

            return hvp_total / float(total_interactions) + self.damping * v

        hvp_total = torch.zeros_like(v)
        total_samples = 0

        for start_idx in range(0, num_active, self.hvp_batch_size):
            end_idx = min(start_idx + self.hvp_batch_size, num_active)
            current_batch_size = end_idx - start_idx
            indices = active_indices[start_idx:end_idx]

            user_ids = [self.full_train_data[i][0] for i in indices]
            pos_items = [self.full_train_data[i][1] for i in indices]
            neg_items = [self.full_train_data[i][2] for i in indices]

            batch_loss = self._batch_train_loss(user_ids, pos_items, neg_items)
            grad = self._grad(batch_loss, params)

            hvp_batch = torch.autograd.grad(grad, params, grad_outputs=v, retain_graph=True, allow_unused=True)
            hvp_batch = [h if h is not None else torch.zeros_like(p) for h, p in zip(hvp_batch, params)]
            hvp_total += self._flatten_grads(hvp_batch) * current_batch_size
            total_samples += current_batch_size

        return hvp_total / float(total_samples) + self.damping * v

    def _hvp_sliced(self, params, v_sub, target_item, train_indices=None):
        if self.full_train_data is None or len(self.full_train_data) == 0:
            return self.damping * v_sub

        active_indices = train_indices if train_indices is not None and len(train_indices) > 0 else np.arange(len(self.full_train_data))
        num_active = len(active_indices)
        if num_active == 0:
            return self.damping * v_sub

        hvp_total = torch.zeros_like(v_sub)
        total_samples = 0

        for start_idx in range(0, num_active, self.hvp_batch_size):
            end_idx = min(start_idx + self.hvp_batch_size, num_active)
            current_batch_size = end_idx - start_idx
            indices = active_indices[start_idx:end_idx]

            user_ids = [self.full_train_data[i][0] for i in indices]
            pos_items = [self.full_train_data[i][1] for i in indices]
            neg_items = [self.full_train_data[i][2] for i in indices]

            batch_loss = self._batch_train_loss(user_ids, pos_items, neg_items)

            grad_all = torch.autograd.grad(batch_loss, params, create_graph=True, retain_graph=True, allow_unused=True)
            grad_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad_all, params)]

            grad_sub = self._slice_grads(grad_all, params, target_item)

            dot = torch.dot(grad_sub, v_sub.detach())

            grad2_all = torch.autograd.grad(dot, params, retain_graph=True, allow_unused=True)
            grad2_all = [g if g is not None else torch.zeros_like(p) for g, p in zip(grad2_all, params)]
            hvp_sub = self._slice_grads(grad2_all, params, target_item)

            hvp_total += hvp_sub.detach() * current_batch_size
            total_samples += current_batch_size

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

    def _get_train_indices_of_test_case(self, user_id, target_item):
        user_indices = self.user_to_train_indices.get(user_id, [])
        item_indices = self.item_to_train_indices.get(target_item, [])
        combined = set(user_indices) | set(item_indices)
        return np.array(sorted(combined)) if combined else np.array([], dtype=int)
'''
Based on the original implementation of paper "Joint Factual and Counterfactual Explanations for Top-k GNN-based Recommendations"
Original repository: https://github.com/Mewtwo1996/GREASE
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
import numpy as np
from .base_model import GraphExpBaseModel

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, with_bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if with_bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SurrogateGCN(nn.Module):
    def __init__(self, nfeat, nhid, dims, num_nodes, dropout=0.5, with_relu=True, with_bias=True, device=None):
        super(SurrogateGCN, self).__init__()
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.dims = dims
        self.num_nodes = num_nodes
        
        self.gc1 = GraphConvolution(nfeat, nhid, with_bias=with_bias)
        self.gc2 = GraphConvolution(nhid, dims, with_bias=with_bias) 
        self.dropout = dropout
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.sub_adj = None
        self.P_hat_symm = None
        
        self.gc1.weight.requires_grad = True
        self.gc2.weight.requires_grad = True
        self.reset_parameters()

    def reset_parameters(self):
        self.P_vec_size = int((self.num_nodes * self.num_nodes - self.num_nodes) / 2) + self.num_nodes
        self.P_vec = Parameter(torch.ones(self.P_vec_size, device=self.device, dtype=torch.float32))

    def _create_symm_matrix_from_vec(self, vector):
        matrix = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
        idx = torch.tril_indices(self.num_nodes, self.num_nodes, device=self.device)
        matrix[idx[0], idx[1]] = vector
        return torch.tril(matrix) + torch.tril(matrix, -1).t()

    def forward(self, x, adj):
        self.sub_adj = adj
        self.P_hat_symm = self._create_symm_matrix_from_vec(self.P_vec)

        A_tilde = torch.sigmoid(self.P_hat_symm) * self.sub_adj + torch.eye(self.num_nodes, device=self.device)
        deg = A_tilde.sum(dim=1).detach()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_adj = deg_inv_sqrt.unsqueeze(1) * A_tilde * deg_inv_sqrt.unsqueeze(0)
        
        if self.with_relu:
            x = F.relu(self.gc1(x, norm_adj))
        else:
            x = self.gc1(x, norm_adj)
        x = F.dropout(x, self.dropout, training=self.training)
        
        x = self.gc2(x, norm_adj)
        return x


class GREASE(GraphExpBaseModel):
    def __init__(self, rec_model, device, args, config=None):
        super().__init__(rec_model, device, args, config)
        self.n_users = rec_model.n_users
        self.n_items = rec_model.n_items
        self.mode = 'explicit'
        self.surrogate_epochs = config.get('surrogate_epochs', 500) 
        self.surrogate_lr = config.get('surrogate_lr', 0.01)
        self.perturb_epochs = config.get('perturb_epochs', 300)
        self.perturb_lr = config.get('perturb_lr', 0.1)
        self.alpha_grease = config.get('alpha_grease', 0.001)
        self.nhid = config.get('nhid', 800)

    def _get_rec_model_embeddings(self, rec_model):
        norm_adj = rec_model.ori_norm_adj_mat
        if rec_model.__class__.__name__ == "GFormer":
            return rec_model.propagate(norm_adj, norm_adj, norm_adj, is_test=True)
        return rec_model.propagate(norm_adj)

    def _extract_surrogate(self, user_id, rec_model, target_items=None, graph_perturb="khop"):
        subgraph_mat = self.get_historical_interactions(user_id, target_items, graph_perturb)
        if subgraph_mat.is_sparse:
            subgraph_mat = subgraph_mat.to_dense()

        edge_indices = subgraph_mat.nonzero(as_tuple=False)
        valid_u = edge_indices[:, 0]
        valid_i = edge_indices[:, 1]

        sorted_users = torch.unique(valid_u).cpu().numpy()
        sorted_items = torch.unique(valid_i).cpu().numpy()

        sorted_users = np.union1d(sorted_users, [user_id])
        if target_items is not None:
            sorted_items = np.union1d(sorted_items, np.array(target_items))

        n_users_sub = len(sorted_users)
        n_items_sub = len(sorted_items)
        n_sub_nodes = n_users_sub + n_items_sub

        valid_u_np = valid_u.cpu().numpy()
        valid_i_np = valid_i.cpu().numpy()

        local_u = np.searchsorted(sorted_users, valid_u_np)
        local_i = np.searchsorted(sorted_items, valid_i_np) + n_users_sub

        target_local = np.searchsorted(sorted_users, [user_id])[0]

        target_item_local_indices = []
        if target_items is not None:
            target_item_local_indices = (np.searchsorted(sorted_items, target_items) + n_users_sub).tolist()

        user_ids_t = torch.tensor(sorted_users, dtype=torch.long, device=self.device)
        item_ids_t = torch.tensor(sorted_items, dtype=torch.long, device=self.device)

        if hasattr(rec_model, 'E0'):
            user_emb_0 = rec_model.E0(user_ids_t)
            item_emb_0 = rec_model.E0(item_ids_t + self.n_users)
        elif hasattr(rec_model, 'uEmbeds') and hasattr(rec_model, 'iEmbeds'):
            user_emb_0 = rec_model.uEmbeds[user_ids_t]
            item_emb_0 = rec_model.iEmbeds[item_ids_t]
        else:
            user_emb_0 = rec_model.embedding_user(user_ids_t)
            item_emb_0 = rec_model.embedding_item(item_ids_t)

        features = torch.cat([user_emb_0, item_emb_0], dim=0)

        with torch.no_grad():
            full_u, full_i = self._get_rec_model_embeddings(rec_model)
            sub_u_emb = full_u[user_ids_t]
            sub_i_emb = full_i[item_ids_t]
            target_embeddings = torch.cat([sub_u_emb, sub_i_emb], dim=0)

        sub_adj = torch.zeros(n_sub_nodes, n_sub_nodes, device=self.device)
        local_u_t = torch.tensor(local_u, dtype=torch.long, device=self.device)
        local_i_t = torch.tensor(local_i, dtype=torch.long, device=self.device)
        sub_adj[local_u_t, local_i_t] = 1.0
        sub_adj[local_i_t, local_u_t] = 1.0

        return features.detach(), sub_adj, target_embeddings.detach(), sorted_items, target_local, n_users_sub, target_item_local_indices, sorted_users

    def _train_surrogate(self, features, sub_adj_dense, target_embeddings):
        num_nodes = features.shape[0]
        feat_dim = features.shape[1]
        
        surrogate = SurrogateGCN(
            nfeat=feat_dim, nhid=self.nhid, dims=feat_dim, 
            num_nodes=num_nodes,
            dropout=0.8, 
            with_relu=True, with_bias=True, device=self.device
        ).to(self.device)
        
        gcn_params = [p for n, p in surrogate.named_parameters() if 'P_vec' not in n]
        optimizer = torch.optim.Adam(gcn_params, lr=self.surrogate_lr, weight_decay=5e-4)
        
        surrogate.train()
        for _ in range(self.surrogate_epochs):
            optimizer.zero_grad()
            output = surrogate(features, sub_adj_dense)
            loss = F.mse_loss(output, target_embeddings)

            loss.backward()
            optimizer.step()

        return surrogate

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        if not isinstance(item_ids, list):
            item_ids = [item_ids]
            
        top_k = self.args.top_k
        graph_perturb = kwargs.get("graph_perturb", "khop")
        cf_results = []
        
        for target_item in item_ids:
            target_items_list = [target_item]

            features, sub_adj_dense, target_embeddings, sub_items_original_ids, target_user_local_idx, n_users_sub, target_item_local_indices, sorted_users = self._extract_surrogate(user_id, self.rec_model, target_items_list, graph_perturb=graph_perturb)
            surrogate = self._train_surrogate(
                features, sub_adj_dense, target_embeddings
            )

            optimizer_p = torch.optim.SGD([surrogate.P_vec], lr=self.perturb_lr)

            for name, param in surrogate.named_parameters():
                if name.endswith("weight") or name.endswith("bias"):
                    param.requires_grad = False

            record = set()
            flag = 0
            old_rank = 0
            batch_users_t = torch.tensor([user_id], device=self.device, dtype=torch.long)
            target_item_tensor = torch.tensor(target_items_list, device=self.device, dtype=torch.long)
            full_mask = torch.ones_like(self.rec_model.ui_mat)

            with torch.no_grad():
                _, rating_K_indices = self.rec_model.predict(users=batch_users_t, topk=max(20, top_k))
                rating_topk = rating_K_indices.squeeze(0)
                init_pos = torch.nonzero(torch.isin(rating_topk, target_item_tensor), as_tuple=False)
                if init_pos.numel() > 0:
                    old_rank = int(init_pos[0, 0].item())

            for _ in range(self.perturb_epochs):
                if flag == 1:
                    break

                output = surrogate(features, sub_adj_dense)
                P = (torch.sigmoid(surrogate.P_hat_symm) >= 0.5).float()
                score = sum(torch.dot(output[target_user_local_idx], output[idx]) for idx in target_item_local_indices)

                cf_adj = P * sub_adj_dense
                dist_loss = torch.abs(cf_adj - sub_adj_dense).sum() / 2

                loss = score + self.alpha_grease * dist_loss
                loss.backward()

                torch.nn.utils.clip_grad_norm_([surrogate.P_vec], 10.0)
                optimizer_p.step()

                with torch.no_grad():
                    removed_indices = ((P.detach() - 1) * sub_adj_dense).nonzero(as_tuple=False)

                    if len(removed_indices) > 0:
                        final_indices = removed_indices[
                            (removed_indices[:, 0] < removed_indices[:, 1])
                            & (removed_indices[:, 0] < n_users_sub)
                            & (removed_indices[:, 1] >= n_users_sub)
                        ]
                        if len(final_indices) == 0:
                            continue

                        for pair in final_indices:
                            u_sub = int(pair[0].item())
                            v_sub = int(pair[1].item())

                            if u_sub >= len(sorted_users): continue
                            if (v_sub - n_users_sub) >= len(sub_items_original_ids): continue

                            uid_global = int(sorted_users[u_sub])
                            iid_global = int(sub_items_original_ids[v_sub - n_users_sub])

                            edge_tuple = (uid_global, iid_global)
                            if edge_tuple in record: continue

                            record.add(edge_tuple)
                            full_mask[uid_global, iid_global] = 0.0

                            _, rating_K_indices = self.rec_model.predict(users=batch_users_t, topk=max(20, top_k), mask=full_mask)
                            rating_topk = rating_K_indices.squeeze(0)
                            target_in_topk = torch.isin(rating_topk[:top_k], target_item_tensor).any().item()
                            if not target_in_topk:
                                flag = 1
                                break

                            new_pos = torch.nonzero(torch.isin(rating_topk, target_item_tensor), as_tuple=False)
                            new_rank = int(new_pos[0, 0].item()) if new_pos.numel() > 0 else int(rating_topk.numel())

                            if new_rank < old_rank:
                                record.discard(edge_tuple)
                                full_mask[uid_global, iid_global] = 1.0

                            old_rank = max(old_rank, new_rank)
            
            cf_results.extend(list(record))

        if len(cf_results) == 0:
            return torch.zeros_like(self.rec_model.ui_mat)
        users, items = zip(*cf_results)
        cf_tensor = torch.tensor([list(users), list(items)], device=self.device, dtype=torch.long)
        return self.convert_cf_list_to_mask(cf_tensor)

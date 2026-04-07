'''
Based on the original implementation of paper "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation"
Original repository: https://github.com/gusye1234/LightGCN-PyTorch
'''

import torch
import torch.nn as nn
from .base_model import GraphRecBaseModel

class LightGCN(GraphRecBaseModel):
    def __init__(self, data_handler, device, args, config, **kwargs):
        super(LightGCN, self).__init__(data_handler, device, args, config, **kwargs)
        self.n_layers = getattr(config, "n_layers", 3)
        self.latent_dim = getattr(config, "latent_dim", 64)
        self.decay = getattr(config, 'decay', 1e-4)
        self.init_embedding()

        self.ori_norm_adj_mat = self.get_A_tilde(self.ui_mat)

    def init_embedding(self):
        self.E0 = nn.Embedding(self.n_users + self.n_items, self.latent_dim).to(self.device)
        nn.init.xavier_uniform_(self.E0.weight)
        self.E0.weight = nn.Parameter(self.E0.weight).to(self.device)

    def get_A_tilde(self, ui_mat, mask=None):
        if ui_mat.is_sparse:
            return self._build_norm_adj_sparse(ui_mat)
        else:
            return self._build_norm_adj_dense(ui_mat, mask=mask)
    
    def _build_norm_adj_dense(self, ui_mat, mask=None):
        if mask != None:
            ui_mat = self.apply_mask(ui_mat, mask)
        else:
            ui_mat = self.ui_mat
        zero_uu = torch.zeros((self.n_users, self.n_users), device=self.device)
        zero_ii = torch.zeros((self.n_items, self.n_items), device=self.device)
        
        # adjacency matrix
        A = torch.cat(
            [
                torch.cat([zero_uu, ui_mat], dim=1),
                torch.cat([ui_mat.t(), zero_ii], dim=1)
            ],
            dim=0
        )
        
        D = torch.diag(A.sum(dim=1)).detach()
        D_exp = D ** (-0.5)
        D_exp[torch.isinf(D_exp)] = 0.0
        A_tilde = D_exp @ A @ D_exp
        
        return A_tilde

    def _build_norm_adj_sparse(self, ui_mat):
        ui_mat = ui_mat.coalesce()
        indices = ui_mat.indices()
        values = ui_mat.values()

        user_idx = indices[0]
        item_idx = indices[1]

        UR = torch.stack([user_idx, item_idx + self.n_users], dim=0)
        BL = torch.stack([item_idx + self.n_users, user_idx], dim=0)

        adj_indices = torch.cat([UR, BL], dim=1)
        adj_values = torch.cat([values, values], dim=0)

        adj_mat = torch.sparse_coo_tensor(
            adj_indices, adj_values,
            size=(self.n_users + self.n_items, self.n_users + self.n_items),
            device=self.device
        ).coalesce()

        rowsum = torch.sparse.sum(adj_mat, dim=1).to_dense()
        d_inv = torch.pow(rowsum + 1e-9, -0.5)
        d_inv[torch.isinf(d_inv)] = 0.0

        diag_idx = torch.arange(len(d_inv), device=self.device)
        d_mat_inv = torch.sparse_coo_tensor(
            torch.vstack([diag_idx, diag_idx]), d_inv,
            size=(len(d_inv), len(d_inv)), device=self.device
        )

        norm_adj_mat = torch.sparse.mm(d_mat_inv, adj_mat)
        norm_adj_mat = torch.sparse.mm(norm_adj_mat, d_mat_inv)
        return norm_adj_mat

    def propagate(self, norm_adj_mat):

        all_layer_embedding = [self.E0.weight]

        for _ in range(self.n_layers):
            if norm_adj_mat.is_sparse:
                E_lyr = torch.sparse.mm(norm_adj_mat.to(self.device), all_layer_embedding[-1])
            else:
                E_lyr = torch.mm(norm_adj_mat.to(self.device), all_layer_embedding[-1])
            all_layer_embedding.append(E_lyr)

        all_layer_embedding = torch.stack(all_layer_embedding)
        mean_layer_embedding = torch.mean(all_layer_embedding, axis=0)

        final_user_embed, final_item_embed = torch.split(mean_layer_embedding, [self.n_users, self.n_items])

        return final_user_embed, final_item_embed

    def forward(self, users, pos_items, neg_items):
        final_user_embed, final_item_embed = self.propagate(self.ori_norm_adj_mat)
        users_emb, pos_emb, neg_emb = final_user_embed[users], final_item_embed[pos_items], final_item_embed[neg_items]

        initial_user_embed, initial_item_embed = self.E0.weight[:self.n_users], self.E0.weight[self.n_users:]
        userEmb0, posEmb0, negEmb0 = initial_user_embed[users], initial_item_embed[pos_items], initial_item_embed[neg_items]

        return self.bpr_loss(users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0)

    def bpr_loss(self, users, users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0, **kwargs):
        reg_loss = (1/2)*(userEmb0.norm().pow(2) +
                        posEmb0.norm().pow(2)  +
                        negEmb0.norm().pow(2))/float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss + self.decay * reg_loss

    def compute_scores(self, users, mask=None):
        if mask != None:
            norm_adj_mat = self.get_A_tilde(self.ui_mat, mask=mask)
        else:
            norm_adj_mat = self.ori_norm_adj_mat
        u_emb, i_emb = self.propagate(norm_adj_mat=norm_adj_mat)
        if users != None:
            u_emb = u_emb[users]
        if isinstance(users, int) or (isinstance(users, list) and len(users) == 1):
            return (u_emb.unsqueeze(0) @ i_emb.T).squeeze()
        return u_emb @ i_emb.T
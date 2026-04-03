import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, ReLU, Linear
from tqdm import tqdm
import math
from .base_model import GraphExpBaseModel
import numpy as np


class C2Explainer(GraphExpBaseModel):
    coeffs = {
        'beta': 1,  # for perturb loss
        'gamma': 0.1,  # for entropy loss
        'EPS': 1e-15,  # epsilon
    }
    
    def __init__(self, model, device, args, config):
        super().__init__(model, device, args, config)
        self.mode = "hybrid"
        self.edge_mask_delete = None
        self.add_edges = config.get("add_edges", False)
        self.k_hop = 3
        self.silent = config.get("silent", True)
        if self.add_edges:
            self.get_graph = self.get_complete_graph 
        else:
            self.get_graph = self.get_graph_edges

        self.top_2k_indices = torch.topk(self.y_pred_scores, 2*self.args.top_k, dim=1)[1]
        self.init_origin_mask()
        
    def init_origin_mask(self, index=None):
        all_user_idx, all_item_idx = self.get_graph(index) 
        n_added_edges = len(all_user_idx)
        if self.add_edges:
            self.orig_mask_delete = torch.zeros(n_added_edges, device=self.device)
            real_user_idx, real_item_idx = self.get_graph_edges(index)
            real_edges = set(zip(real_user_idx.cpu().tolist(), real_item_idx.cpu().tolist()))
            all_edges = list(zip(all_user_idx.cpu().tolist(), all_item_idx.cpu().tolist()))
            for idx, edge in enumerate(all_edges):
                if edge in real_edges:
                    self.orig_mask_delete[idx] = 1.0
        else:
            self.orig_mask_delete = torch.ones(n_added_edges, device=self.device)

    def _train_mask(self, node_idx, item_idx=None):
        parameters = [self.edge_mask_delete]
        optimizer = torch.optim.Adam(parameters, lr=self.config.get("lr", 0.1))
        all_user_idx, all_item_idx = self.get_graph(node_idx) 
        
        orig_mask = self.orig_mask_delete

        cfs = []
        cf_labels = []
        cf_perturbs = []
        
        num_epochs = self.config.get("epochs", 50)
        pbar = tqdm(range(1, num_epochs+1), desc=f"Training node: {node_idx}", disable=self.silent)

        y = self.top_2k_indices[node_idx][self.args.top_k:]
            
        for i in pbar:
            optimizer.zero_grad()
            edge_mask = self._to_edge_mask()
            soft_edge_mask = torch.sigmoid(self.edge_mask_delete)
            
            masked_ui_mat = torch.ones(self.ui_mat.shape, device=self.device)
            preservation_subgraph = 1 - torch.abs(edge_mask - self.ui_mat[all_user_idx, all_item_idx])
            masked_ui_mat[all_user_idx, all_item_idx] = preservation_subgraph

            y_hat = self.rec_model.predict(users=node_idx, mask=masked_ui_mat)

            cf = (edge_mask, soft_edge_mask)
            y_pred_indices = torch.topk(y_hat, self.args.top_k).indices
            y_pred_sorted = y_pred_indices.sort()[0]
            y_sorted = y.sort()[0]
            is_different = not torch.equal(y_pred_sorted, y_sorted)
            if is_different:
                perturb_mat = (edge_mask - orig_mask).detach().clone()
                num_perturb = torch.sum(torch.abs(perturb_mat)).item()
                cfs.append(cf)
                cf_perturbs.append(num_perturb)
                cf_labels.append(torch.argmax(y_hat).item())

            cf_loss = self.loss(y_hat, y)
            perturb_loss = self.perturb_loss()
            loss = cf_loss + self.coeffs['beta'] * perturb_loss
            loss.backward()   
            optimizer.step()    
            pbar.set_postfix({
                'loss': f'{loss:.4f}',
                'cf_loss': f'{cf_loss:.4f}',
                'perturb': f'{perturb_loss:.4f}',
            })           
        return cfs, cf_labels, cf_perturbs

    def _run_explain(self, user_id):
        self.init_mask(user_id)
        all_user_idx, all_item_idx = self.get_graph(user_id)
        cfs, cf_labels, cf_perturbs = self._train_mask(user_id)
        best_cf, best_cf_soft = self._post_process_cfs(cfs, cf_perturbs)

        final_mask = torch.zeros(self.ui_mat.shape, device=self.device)
        final_mask[all_user_idx, all_item_idx] = best_cf

        soft_ui_mat = torch.zeros(self.ui_mat.shape, device=self.device)
        soft_ui_mat[all_user_idx, all_item_idx] = best_cf_soft

        return final_mask, soft_ui_mat

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        final_mask, _ = self._run_explain(user_id)
        final_mask = final_mask * interaction
        orig_mask = self.ui_mat * interaction
        final_mask = torch.abs(final_mask - orig_mask)
        return final_mask

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        _, soft_ui_mat = self._run_explain(user_id)
        importance_score = 1 - soft_ui_mat
        importance_score = importance_score * interaction
        importance_score[(importance_score == 0) & (interaction > 0)] = 1e-9
        return importance_score
    
    def init_mask(self, index=None):
        all_user_idx, all_item_idx = self.get_graph(index)
        n_added_edges = len(all_user_idx)       
        edge_mask_delete = torch.rand(n_added_edges, device=self.device) * 2 - 1 
        self.edge_mask_delete = nn.Parameter(edge_mask_delete)
    
    def loss(self, pred, pred_label):
        log_probs = F.log_softmax(pred, dim=-1)
        loss = -log_probs[pred_label].mean()
        return loss
    
    def perturb_loss(self):
        a = torch.sigmoid(self.edge_mask_delete) - self.orig_mask_delete
        perturb_loss = torch.mean(torch.abs(a))
        return perturb_loss
    
    def _to_edge_mask(self): 
        edge_mask_delete = torch.sigmoid(self.edge_mask_delete) 
        return self._ST_trick(edge_mask_delete)

    def _ST_trick(self, edge_mask):
        binarized_edge_mask = (edge_mask > 0.5).detach().clone().to(torch.int)
        edge_mask = binarized_edge_mask - edge_mask.detach() + edge_mask
        return edge_mask
    
    def _post_process_cfs(self, cfs, cf_perturbs):
        if not cfs:
            return self.ui_mat, self.ui_mat
        else:
            min_perturbs = min(cf_perturbs)
            min_perturb_index = cf_perturbs.index(min_perturbs)
            best_cf_hard, best_cf_soft = cfs[min_perturb_index]  

        self._clean_model()
        return best_cf_hard, best_cf_soft
    
    def _clean_model(self):
        self.edge_mask_delete = None

    def get_graph_edges(self, node_idx=None):
        user_idx, item_idx = torch.where(self.ui_mat != 0)
        return user_idx, item_idx
    
    def get_complete_graph(self, node_idx=None):
        user_idx, item_idx = self.get_graph_edges(node_idx)
        unique_users = torch.unique(user_idx).sort()[0]
        unique_items = torch.unique(item_idx).sort()[0]
        user_idx = unique_users.repeat_interleave(len(unique_items))
        item_idx = unique_items.repeat(len(unique_users))
        return user_idx, item_idx
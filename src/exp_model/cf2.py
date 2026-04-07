'''
Based on the original implementation of the paper "Learning and Evaluating Graph Neural Network Explanations based on Counterfactual and Factual Reasoning"
Original repository: https://github.com/chrisjtan/gnn_cff
'''

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import math
from .base_model import GraphExpBaseModel

class CF2(GraphExpBaseModel):
    def __init__(self, model, device, args, config):
        super(CF2, self).__init__(model, device, args, config)
        self.mode = "explicit"
        self.lam = getattr(config, "lam", 500.)
        self.lr = getattr(config, "lr", 0.5)
        self.num_epochs = getattr(config, "num_epochs", 20)
        self.relu = nn.ReLU()

    def _init_mask(self):
        self.mask = torch.nn.Parameter(torch.ones(self.ui_mat.shape, device=self.device))
        std = nn.init.calculate_gain("relu") * math.sqrt(
            2.0 / (self.rec_model.n_users + self.rec_model.n_items)
        )
        with torch.no_grad():
            self.mask.normal_(1.0, std)

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        self._init_mask()
        optimizer = Adam([self.mask], lr=self.lr)
        self.rec_model.eval()
        self.train()        
        pbar = tqdm(range(self.num_epochs))

        for _ in pbar:
            loss_total, L1, bpr = self.train_epoch(optimizer, user_id)
            pbar.set_postfix({"loss": loss_total, "L1_loss": L1, "bpr_loss": bpr})

        mask_actual = (torch.sigmoid(self.mask) >= 0.5).float()
        mask_actual = mask_actual * interaction + 1 - interaction
        mask_actual = torch.clamp(mask_actual, min=0.0, max=1.0)
        return self.flip_mask(mask_actual)

    def train_epoch(self, optimizer, node_idx):
        optimizer.zero_grad()
        mask = torch.sigmoid(self.mask)
        output = self.rec_model.predict(node_idx, mask=mask)
        if not torch.all((output >= 0) & (output <= 1)):
            output = torch.sigmoid(output)
        loss_total, L1, bpr = self.loss(mask, output, self.y_pred_indices[node_idx])
        loss_total.backward()
        optimizer.step()
        return loss_total.item(), L1.item(), bpr.item()
    
    def loss(self, mask, pred, pred_label):
        cf_pred_label = pred[pred_label].min()
        valid_mask = torch.ones_like(pred, dtype=torch.bool)
        valid_mask[pred_label] = False
        cf_next = pred[valid_mask].max()
        
        bpr = self.relu(0.5 + cf_pred_label - cf_next)
        L1 = torch.linalg.norm(torch.abs(1 - mask), ord=1)
        loss_total = L1 + self.lam * bpr
        return loss_total, L1, bpr
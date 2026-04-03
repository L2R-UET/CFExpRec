import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
from .base_model import GraphExpBaseModel

class CFGNNExplainer(GraphExpBaseModel):

    def __init__(self, model, device, args, config):
        super(CFGNNExplainer, self).__init__(model, device, args, config)
        self.mode = "explicit"
        self.beta = getattr(config, "beta", 0.5)
        self.lr = getattr(config, "lr", 1)
        self.num_epochs = getattr(config, "num_epochs", 50)

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        self.mask = nn.Parameter(torch.ones(self.ui_mat.shape, device=self.device))
        
        optimizer = optim.SGD([self.mask], lr=self.lr)
        self.rec_model.eval()
        self.train()
        best_cf_example = self.mask.clone()
        best_loss = float('inf')
        pbar = tqdm(range(self.num_epochs))

        for _ in pbar:
            new_example, loss_total, loss_pred, loss_graph_dist = self.train_epoch(optimizer, user_id, item_ids)
            if new_example is not None and loss_total < best_loss:
                best_cf_example = new_example
                best_loss = loss_total
            pbar.set_postfix({"loss": loss_total, "pred_loss": loss_pred, "dist_loss": loss_graph_dist})

        mask_actual = (torch.sigmoid(best_cf_example) >= 0.5).float()
        mask_actual = mask_actual * interaction + 1 - interaction
        mask_actual = torch.clamp(mask_actual, min=0.0, max=1.0)
        return self.flip_mask(mask_actual)
    
    def train_epoch(self, optimizer, user_id, item_ids):
        optimizer.zero_grad()
        mask = F.sigmoid(self.mask)
        output = self.rec_model.predict(user_id, mask=mask)
        mask_actual = (F.sigmoid(self.mask) >= 0.5).float()
        _, y_pred_actual = self.rec_model.predict(user_id, self.args.top_k, mask=mask_actual)
        y_pred_actual = y_pred_actual.detach()
        new_example = None
        if isinstance(item_ids, list):
            if torch.isin(y_pred_actual, torch.tensor(item_ids, device=self.device)).float().mean() != 1.:
                new_example = self.mask.clone()
        else:
            if item_ids not in self.y_pred_indices[user_id]:
                new_example = self.mask.clone()
            
        loss_total, loss_pred, loss_graph_dist = self.loss(output, self.y_pred_indices[user_id], y_pred_actual, mask_actual, item_ids)
        loss_total.backward()
        clip_grad_norm_(self.parameters(), 2.0)
        optimizer.step()

        return new_example, loss_total.item(), loss_pred.item(), loss_graph_dist.item()
        

    def loss(self, output, y_pred_orig, y_pred_actual, mask_actual, item_id):
        if isinstance(item_id, list):
            pred_same = torch.isin(y_pred_actual, y_pred_orig).float().mean()
            log_probs = F.log_softmax(output, dim=-1)
            loss_pred = log_probs[y_pred_orig].mean()
        else:
            pred_same = (y_pred_actual == item_id).float().sum()
            log_probs = F.log_softmax(output, dim=-1)
            loss_pred = log_probs[item_id]
        cf_adj = self.ui_mat * mask_actual
        cf_adj.requires_grad = True
        loss_graph_dist = torch.sum(torch.abs(cf_adj - self.ui_mat))
        loss_total = pred_same * loss_pred + self.beta * loss_graph_dist
        return loss_total, loss_pred, loss_graph_dist
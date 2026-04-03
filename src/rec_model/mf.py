import torch
import torch.nn as nn
import torch.nn.functional as F
from rec_model.base_model import UserVectorRecBaseModel

class MF(UserVectorRecBaseModel):
    def __init__(self, data_handler, device, args, config=None, **kwargs):
        super(MF, self).__init__(data_handler, device, args, config, **kwargs)
        self.latent_dim = config.get('latent_dim', 64)
        self.users_fc = nn.Linear(self.n_items, self.latent_dim, bias=True)
        self.item_fc = nn.Embedding(self.n_items, self.latent_dim)
        self.item_bias = nn.Parameter(torch.zeros(self.latent_dim))
        self.reg_lambda = config.get('lambda_2', 1e-5)
        self.to(device)

    def forward(self, users, pos_items, neg_items):
        batch_history = self.get_interaction_vectors(users)
        user_vec = self.users_fc(batch_history)
        
        pos_vec = self.item_fc(pos_items) + self.item_bias
        neg_vec = self.item_fc(neg_items) + self.item_bias
        
        pos_scores = (user_vec * pos_vec).sum(dim=1)
        neg_scores = (user_vec * neg_vec).sum(dim=1)
        
        pos_scores = torch.sigmoid(pos_scores)
        neg_scores = torch.sigmoid(neg_scores)
        
        pos_labels = torch.ones_like(pos_scores)
        neg_labels = torch.zeros_like(neg_scores)
        
        loss = F.binary_cross_entropy(pos_scores, pos_labels) + \
               F.binary_cross_entropy(neg_scores, neg_labels)
               
        reg_loss = (user_vec.norm(2).pow(2) + pos_vec.norm(2).pow(2) + neg_vec.norm(2).pow(2)) / 2
        return loss + self.reg_lambda * reg_loss / len(users)

    def compute_scores(self, users, mask=None):
        if users is None:
            users = torch.arange(self.n_users, device=self.device)
        batch_history = self.get_interaction_vectors(users)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.stack(mask)
            batch_history = self.apply_mask(batch_history, mask)
        user_vec = self.users_fc(batch_history)
        all_item_vec = self.item_fc.weight + self.item_bias
        return torch.matmul(user_vec, all_item_vec.T)
'''
Based on the implementation in the LXR repository: https://github.com/DeltaLabTLV/LXR
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from .base_model import UserVectorRecBaseModel

class VAE(UserVectorRecBaseModel):
    def __init__(self, data_handler, device, args, config=None, **kwargs):
        super(VAE, self).__init__(data_handler, device, args, config, **kwargs)
        
        self.enc_dims = config.get('enc_dims', [512, 128])
        self.dec_dims = config.get('dec_dims', [128, 512])

        if isinstance(self.enc_dims, str):
            self.enc_dims = json.loads(self.enc_dims)
        if isinstance(self.dec_dims, str):
            self.dec_dims = json.loads(self.dec_dims)
        
        if self.enc_dims[0] != self.n_items:
            self.enc_dims = [self.n_items] + self.enc_dims
            
        if self.dec_dims[-1] != self.n_items:
            self.dec_dims = self.dec_dims + [self.n_items]

        self.dropout = config.get('dropout', 0.5)
        if self.dropout is None:
            self.dropout = 0.5
        self.anneal_cap = config.get('anneal_cap', 0.2)
        self.total_anneal_steps = config.get('total_anneal_steps', 200000)
        
        self.update_count = 0
        
        self.encoder_layers = nn.ModuleList()
        for i in range(len(self.enc_dims) - 1):
            layer = nn.Linear(self.enc_dims[i], self.enc_dims[i+1])
            self.encoder_layers.append(layer)
            
        self.fc_mu = nn.Linear(self.enc_dims[-1], self.enc_dims[-1])
        self.fc_logvar = nn.Linear(self.enc_dims[-1], self.enc_dims[-1])
        
        self.decoder_layers = nn.ModuleList()
        for i in range(len(self.dec_dims) - 1):
             layer = nn.Linear(self.dec_dims[i], self.dec_dims[i+1])
             self.decoder_layers.append(layer)
            
        self.to(device)

    def encode(self, x):
        h = F.normalize(x, dim=-1)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        for layer in self.encoder_layers:
            h = layer(h)
            h = F.relu(h)
            
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        h = z
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
            if i != len(self.decoder_layers) - 1:
                h = F.relu(h)
        return h

    def forward(self, users, pos_items, neg_items):
        batch_ratings = self.get_interaction_vectors(users)
        
        mu, logvar = self.encode(batch_ratings)
        z = self.reparameterize(mu, logvar)
        recon_logits = self.decode(z)
        
        log_softmax_var = F.log_softmax(recon_logits, dim=1)
        neg_ll = -torch.sum(log_softmax_var * batch_ratings, dim=1)
        
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        
        if self.total_anneal_steps > 0:
            self.update_count += 1
            anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
        else:
            anneal = self.anneal_cap
            
        loss = torch.mean(neg_ll + anneal * kl_div)
        
        return loss

    def compute_scores(self, users, mask=None):
        if users is None:
            users = torch.arange(self.n_users, device=self.device)
        batch_history = self.get_interaction_vectors(users)
        if mask is not None:
            if not torch.is_tensor(mask):
                mask = torch.stack(mask)
            batch_history = self.apply_mask(batch_history, mask)
        mu, _ = self.encode(batch_history)
        return torch.softmax(self.decode(mu), dim=1)

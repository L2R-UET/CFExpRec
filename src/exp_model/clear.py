'''
Based on the original implementation of paper "CLEAR: Generative Counterfactual Explanations on Graphs".
Original repository: https://github.com/jma712/GraphCFE
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .base_model import GraphExpBaseModel
from numbers import Number
from metrics.exp_metrics import pn_s_list_one_instance


class CLEAR(GraphExpBaseModel):
    def __init__(self, model, device, args, config):
        super(CLEAR, self).__init__(model, device, args, config)
        self.mode = "hybrid"
        self.h_dim = config.get("latent_dim", 64)
        self.z_dim = config.get("dim_z", 16)
        self.cf_dim = args.top_k
        self.dropout = config.get("dropout", 0.0)
        self.n_items = self.rec_model.n_items
        self.n_users = self.rec_model.n_users
        self.max_num_nodes = self.rec_model.n_users + self.rec_model.n_items
        self.graph_pool_type = 'mean'

        # prior
        self.prior_mean = nn.Linear(1, self.z_dim).to(device)
        self.prior_var = nn.Sequential(nn.Linear(1, self.z_dim), nn.Sigmoid()).to(device)
        self.top_2k_indices = torch.topk(self.y_pred_scores, 2*self.args.top_k, dim=1)[1]

        # encoder
        self.encoder_mean = nn.Sequential(nn.Linear(self.h_dim + 1 + self.cf_dim, self.z_dim), nn.ReLU()).to(device)
        self.encoder_var = nn.Sequential(nn.Linear(self.h_dim + 1 + self.cf_dim, self.z_dim), nn.ReLU(), nn.Sigmoid()).to(device)

        # decoder
        self.decoder_a = nn.Sequential(
            nn.Linear(self.z_dim + self.cf_dim, self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim), nn.Dropout(self.dropout), nn.ReLU(),
            nn.Linear(self.h_dim, self.n_users*self.n_items), nn.Sigmoid()
        ).to(device)
        
        import os
        current_dir = os.getcwd()
        if '/kaggle/input' in current_dir:
            checkpoint_dir = '/kaggle/working/checkpoints'
        else:
            checkpoint_dir = 'checkpoints'
        self.save_path = f'{checkpoint_dir}/clear_{self.args.dataset}_{self.args.top_k}_best_model.pt'
        self.train_model()

    def encoder(self, u, y_cf, mask=None):
        if mask is not None:
            preservation_mask = 1 - torch.abs(mask - self.rec_model.ui_mat)
            norm_adj_mat = self.rec_model.get_A_tilde(self.rec_model.ui_mat, mask=preservation_mask)
            user_emb, item_emb = self.rec_model.propagate(norm_adj_mat)
        else:
            user_emb, item_emb = self.rec_model.propagate(self.rec_model.ori_norm_adj_mat)
        node_emb = torch.cat([user_emb, item_emb], dim=0)
        graph_rep = torch.mean(node_emb, dim=0)
        u_flat = u.flatten()
        encoder_input = torch.cat([graph_rep, u_flat, y_cf])
        z_mu = self.encoder_mean(encoder_input)
        z_logvar = self.encoder_var(encoder_input)
        return z_mu, z_logvar

    def get_represent(self, u, y_cf, mask=None):
        z_mu, z_logvar = self.encoder(u, y_cf, mask=mask)
        return z_mu, z_logvar

    def decoder(self, z, y_cf):
        adj_reconst = self.decoder_a(torch.cat((z, y_cf))).view(self.n_users, self.n_items)
        return adj_reconst

    def graph_pooling(self, x, type='mean'):
        if type == 'max':
            out, _ = torch.max(x, dim=1, keepdim=False)
        elif type == 'sum':
            out = torch.sum(x, dim=1, keepdim=False)
        elif type == 'mean':
            out = torch.mean(x, dim=1, keepdim=False)
        return out

    def prior_params(self, u):  # P(Z|U)
        u_flat = u.flatten() 
        z_u_logvar = self.prior_var(u_flat)  # [z_dim]
        z_u_mu = self.prior_mean(u_flat)  # [z_dim]
        return z_u_mu, z_u_logvar

    def reparameterize(self, mu, logvar):
        '''
        compute z = mu + std * epsilon
        '''
        if self.training:
            # compute the standard deviation from logvar
            std = torch.exp(0.5 * logvar)
            # sample epsilon from a normal distribution with mean 0 and
            # variance 1
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, u, y_cf):
        z_u_mu, z_u_logvar = self.prior_params(u)
        z_mu, z_logvar = self.encoder(u, y_cf)
        z_sample = self.reparameterize(z_mu, z_logvar)
        adj_reconst = self.decoder(z_sample, y_cf)

        return {'z_mu': z_mu, 'z_logvar': z_logvar,
                'adj_reconst': adj_reconst, 'z_u_mu': z_u_mu, 'z_u_logvar': z_u_logvar}

    def train_model(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.config.get("lr", 0.001), 
                                     weight_decay=self.config.get("weight_decay", 1e-5))
        alpha = 5
        cfe_activated = False
        sim_threshold = 0.15
        kl_threshold = 1.0
        u_num = 10
        best_val_loss = float('inf') 
        epochs = self.config.get("epochs", 100)
        
        for epoch in range(epochs):
            epoch_loss, epoch_loss_kl, epoch_loss_sim, epoch_loss_cfe, epoch_loss_kl_cf = 0.0, 0.0, 0.0, 0.0, 0.0
            batch_num = 0
            
            for batch_idx, user_idx in enumerate(self.rec_model.data_handler.train_loader):
                optimizer.zero_grad()
                
                user_idx = user_idx.item()

                u = torch.randint(0, u_num, (1, 1)).float().to(self.device)

                y_cf = self.top_2k_indices[user_idx][self.args.top_k:]
                model_return = self.forward(u, y_cf)

                z_mu_cf, z_logvar_cf = self.get_represent(u, y_cf, model_return['adj_reconst'])
                kl_loss = self.kl_loss(model_return['z_u_logvar'], model_return['z_logvar'], 
                                      model_return['z_mu'], model_return['z_u_mu'])
                
                sim_loss = self.similarity_loss(model_return['adj_reconst'])

                preservation_mask = 1 - torch.abs(model_return['adj_reconst'] - self.rec_model.ui_mat)
                y_pred = self.rec_model.predict(user_idx, mask=preservation_mask)
                cfe_loss = self.CFE_loss(y_pred, y_cf)

                loss_kl_cf = self.rep_loss(model_return['z_mu'], z_mu_cf, z_logvar_cf, model_return['z_logvar'])
                
                if not cfe_activated and sim_loss.item() < sim_threshold and kl_loss.item() < kl_threshold:
                    cfe_activated = True
                    print(f"CFE Loss Activated at Epoch {epoch} (sim_loss={sim_loss.item():.4f}, kl_loss={kl_loss.item():.4f})")
                
                if cfe_activated:
                    loss_batch = sim_loss + kl_loss + alpha * cfe_loss
                else:
                    loss_batch = sim_loss + kl_loss
                
                loss_batch.backward()
                optimizer.step()
                
                epoch_loss += loss_batch.item()
                epoch_loss_kl += kl_loss.item()
                epoch_loss_sim += sim_loss.item()
                epoch_loss_cfe += cfe_loss.item()
                epoch_loss_kl_cf += loss_kl_cf.item() if isinstance(loss_kl_cf, torch.Tensor) else loss_kl_cf
                batch_num += 1
                if batch_idx % 100 == 0:
                    print(f"Epoch {epoch}, User {batch_idx}, Loss: {loss_batch.item():.4f}")
            
            avg_loss = epoch_loss / batch_num
            avg_loss_kl = epoch_loss_kl / batch_num
            avg_loss_sim = epoch_loss_sim / batch_num
            avg_loss_cfe = epoch_loss_cfe / batch_num
            avg_loss_kl_cf = epoch_loss_kl_cf / batch_num

            print(f"Epoch {epoch}: Loss={avg_loss:.4f} || KL={avg_loss_kl:.4f} || Sim={avg_loss_sim:.4f} || CFE={avg_loss_cfe:.4f} || KL_CF={avg_loss_kl_cf:.4f}")

            if cfe_activated == True:
                self.eval()
                val_loss = 0
                pn_hit = []
                for batch_idx, user_idx in enumerate(self.rec_model.data_handler.val_loader):
                    user_idx = user_idx.item()
                    u = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
                    
                    y_cf = self.top_2k_indices[user_idx][self.args.top_k:]
                    model_return = self.forward(u, y_cf)
                    z_mu_cf, z_logvar_cf = self.get_represent(u, y_cf, model_return['adj_reconst'])
                    kl_loss = self.kl_loss(model_return['z_u_logvar'], model_return['z_logvar'], 
                                        model_return['z_mu'], model_return['z_u_mu'])
                    sim_loss = self.similarity_loss(model_return['adj_reconst'])
                    preservation_mask = 1 - torch.abs(model_return['adj_reconst'] - self.rec_model.ui_mat)
                    y_pred = self.rec_model.predict(user_idx, mask=preservation_mask)
                    cfe_loss = self.CFE_loss(y_pred, y_cf)

                    loss_kl_cf = self.rep_loss(model_return['z_mu'], z_mu_cf, z_logvar_cf, model_return['z_logvar'])
                    
                    if cfe_activated:
                        val_loss += sim_loss.item() + kl_loss.item() + alpha * cfe_loss.item()
                    else:
                        val_loss += sim_loss.item() + kl_loss.item()
                    
                    preservation_mask = 1 - torch.abs(model_return['adj_reconst'] - self.rec_model.ui_mat)
                    scores = self.rec_model.predict(user_idx, mask=preservation_mask)          
                    y_pred_indices = torch.topk(scores, self.args.top_k)[1]
                    pn_hit.append(
                        pn_s_list_one_instance(
                            self.y_pred_indices[user_idx].tolist(),
                            y_pred_indices.tolist(),
                            self.args.top_k,
                        )
                    )
                avg_val_loss = val_loss / len(self.rec_model.data_handler.val_loader)
                avg_pn_hit = sum(pn_hit) / len(pn_hit)
                
                print(f"Eval pn_hit: {avg_pn_hit:.4f}")
                print(f"Eval loss: {avg_val_loss:.4f}")
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_pn_hit = avg_pn_hit
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': best_val_loss,
                        'pn_hit': best_pn_hit,
                        'cfe_activated': cfe_activated
                    }
                    
                    import os
                    checkpoint_dir = os.path.dirname(self.save_path)
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save(checkpoint, self.save_path)
                    print(f"Saved best model at epoch {epoch} with val_loss={best_val_loss:.4f}, pn_hit={best_pn_hit:.4f}")

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.eval()
        return checkpoint

    def _get_y_cf(self, user_id, item_ids):
        if isinstance(item_ids, list):
            y_cf = self.top_2k_indices[user_id][self.args.top_k:]
        else:
            item_topk = (self.y_pred_indices[user_id] == item_ids).nonzero(as_tuple=True)[0].item()
            y_cf = self.top_2k_indices[user_id][item_topk:item_topk + self.args.top_k]
        return y_cf

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        self.load_checkpoint(self.save_path)
        u = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
        y_cf = self._get_y_cf(user_id, item_ids)
        model_return = self.forward(u, y_cf)
        final_mask = (model_return['adj_reconst'] >= 0.5).float()
        final_mask = torch.abs(final_mask - self.ui_mat)
        final_mask = final_mask * interaction
        return final_mask

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        self.load_checkpoint(self.save_path)
        u = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
        y_cf = self._get_y_cf(user_id, item_ids)
        model_return = self.forward(u, y_cf)
        importance_score = 1 - model_return['adj_reconst']
        importance_score = importance_score * interaction
        importance_score[(importance_score == 0) & (interaction > 0)] = 1e-9
        return importance_score

    def kl_loss(self, z_u_logvar, z_logvar, z_mu, z_u_mu):
        loss_kl = 0.5 * (((z_u_logvar - z_logvar) + ((z_logvar.exp() + (z_mu - z_u_mu).pow(2)) / z_u_logvar.exp())) - 1)
        loss_kl = torch.mean(loss_kl)
        return loss_kl
    
    def distance_graph_prob(self, adj_1, adj_2_prob):
        adj_2_prob = adj_2_prob.clamp(1e-6, 1 - 1e-6)
        dist = F.binary_cross_entropy(adj_2_prob, adj_1)
        return dist

    def similarity_loss(self, adj_reconst):
        return 1.0 * self.distance_graph_prob(self.ui_mat, adj_reconst)

    def CFE_loss(self, y_pred, pred_label):
        log_probs = F.log_softmax(y_pred, dim=-1)
        loss = -log_probs[pred_label].mean()
        return loss
    
    def rep_loss(self, z_mu, z_mu_cf, z_logvar_cf, z_logvar):
        if z_mu_cf is None:
            loss_kl_cf = 0.0
        else:
            loss_kl_cf = 0.5 * (((z_logvar_cf - z_logvar) + ((z_logvar.exp() + (z_mu - z_mu_cf).pow(2)) / z_logvar_cf.exp())) - 1)
            loss_kl_cf = torch.mean(loss_kl_cf)
        return loss_kl_cf
import torch
import numpy as np
import torch.nn as nn
from .base_model import UserVectorExpBaseModel
from pathlib import Path


class LXR(UserVectorExpBaseModel):
    def __init__(self, rec_model, device, args, config=None):
        if config is None:
            config = {}
        hidden_size = config.get('hidden_size', 128)
        super(LXR, self).__init__(rec_model, device, args, config)
        self.n_items = rec_model.n_items
        self.lambda_pos = config.get('lambda_pos', 10.0)
        self.lambda_neg = config.get('lambda_neg', 0.5)
        self.alpha = config.get('alpha_lxr', 1.0)
        self.lr = config.get('lr', 0.001)
        self.epochs = config.get('epochs', 30)
        self.batch_train = config.get('batch_train', args.batch_train if hasattr(args, 'batch_train') else 1)
        self.patience_limit = config.get('patience', 6)

        self.users_fc = nn.Linear(self.n_items, hidden_size)
        self.items_fc = nn.Linear(self.n_items, hidden_size)
        self.bottleneck = nn.Sequential(
            nn.Tanh(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.n_items),
            nn.Sigmoid()
        )
        self.to(device)
        self.mode = 'hybrid'
        self._trained = False

        project_root = Path(__file__).resolve().parent.parent.parent
        ckpt_dir = project_root / 'checkpoints'
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        dataset = args.dataset if hasattr(args, 'dataset') else 'default'
        rec_name = args.rec_model if hasattr(args, 'rec_model') else 'rec'
        self._ckpt_path = str(ckpt_dir / f"LXR_{rec_name}_{dataset}.pth")

    def forward(self, user_interactions, target_item_tensor):
        user_output = self.users_fc(user_interactions.float())
        item_output = self.items_fc(target_item_tensor.float())
        combined_output = torch.cat((user_output, item_output), dim=-1)
        return self.bottleneck(combined_output)

    def compute_loss(self, user_interactions, items_ids, expl_scores, rec_model, users):
        pos_mask = user_interactions * expl_scores
        neg_mask = user_interactions * (1.0 - expl_scores)

        pos_corrected = user_interactions * pos_mask + (1 - user_interactions) * (1 - pos_mask)
        neg_corrected = user_interactions * neg_mask + (1 - user_interactions) * (1 - neg_mask)

        probs_pos = rec_model.predict(users=users, mask=pos_corrected)
        probs_neg = rec_model.predict(users=users, mask=neg_corrected)

        probs_pos = torch.clamp(probs_pos, 1e-7, 1.0 - 1e-7)
        probs_neg = torch.clamp(probs_neg, 1e-7, 1.0 - 1e-7)

        batch_indices = torch.arange(len(items_ids), device=items_ids.device)
        target_probs_pos = probs_pos[batch_indices, items_ids]
        target_probs_neg = probs_neg[batch_indices, items_ids]

        pos_loss = -torch.mean(torch.log(target_probs_pos + 1e-9))
        neg_loss = torch.mean(torch.log(target_probs_neg + 1e-9))
        l1 = expl_scores[user_interactions > 0].mean()

        return self.lambda_pos * pos_loss + self.lambda_neg * neg_loss + self.alpha * l1

    def fit(self):
        if self._trained:
            return
        if Path(self._ckpt_path).exists():
            print(f"[LXR] Loading checkpoint from {self._ckpt_path}")
            self.load_state_dict(torch.load(self._ckpt_path, map_location=self.device))
            self._trained = True
            return

        print(f"[LXR] No checkpoint found. Auto-training...")
        train_data = self.rec_model.data_handler.train_group_user
        rec_model = self.rec_model

        for p in rec_model.parameters():
            p.requires_grad = False
        rec_model.eval()

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        batch_size = min(self.batch_train, len(train_data)) if self.batch_train > 0 else len(train_data)
        best_loss = float('inf')
        patience_counter = 0

        print(f"Training for {self.epochs} epochs, batch={batch_size}, lr={self.lr}")
        for epoch in range(self.epochs):
            if epoch % 15 == 0 and epoch > 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.1

            self.train()
            losses = []
            indices = np.arange(len(train_data))
            batch_iterator = np.array_split(indices, max(1, len(train_data) // batch_size))

            for batch_idx in batch_iterator:
                batch = train_data.iloc[batch_idx]
                users = torch.tensor(batch['user_id'].values, device=self.device, dtype=torch.long)

                interactions = torch.zeros((len(batch), self.n_items), device=self.device)
                for j, item_ids in enumerate(batch['item_ids'].values):
                    if len(item_ids) > 0:
                        interactions[j, item_ids] = 1.0

                with torch.no_grad():
                    before = rec_model.predict(users=users)
                    for idx, item_ids in enumerate(batch['item_ids'].values):
                        if len(item_ids) > 0:
                            before[idx, item_ids] = -1e9
                    target_indices = torch.topk(before, 1, dim=1).indices[:, 0]

                target_item_tensor = torch.zeros((len(batch), self.n_items), device=self.device)
                target_item_tensor.scatter_(1, target_indices.unsqueeze(1), 1.0)

                expl_scores = self.forward(interactions, target_item_tensor)
                loss = self.compute_loss(interactions, target_indices, expl_scores, rec_model, users)
                losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = np.mean(losses)
            print(f"  Epoch {epoch+1}/{self.epochs}: loss={epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(self.state_dict(), self._ckpt_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience_limit:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break

        if Path(self._ckpt_path).exists():
            self.load_state_dict(torch.load(self._ckpt_path, map_location=self.device))
        else:
            torch.save(self.state_dict(), self._ckpt_path)
        print(f"[LXR] Training complete. Checkpoint: {self._ckpt_path}")
        self._trained = True

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        self.fit()
        user_interaction = self.get_historical_interactions(torch.tensor([user_id], device=self.device), item_ids)
        user_tensor = user_interaction.unsqueeze(0).float()

        if not isinstance(item_ids, (list, np.ndarray)):
            item_ids = [item_ids]

        target_item_tensor = torch.zeros_like(user_tensor)
        target_item_tensor[0, item_ids] = 1.0

        with torch.no_grad():
            expl_scores = self.forward(user_tensor, target_item_tensor)

        scores = expl_scores[0]
        history_mask = user_tensor[0] > 0
        scores_tensor = scores * history_mask.float()
        scores_tensor += user_interaction * 1e-12
        
        return scores_tensor

    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        self.fit()
        if not isinstance(item_ids, list):
            item_ids = [item_ids]
            
        cf_results = []
        for target_item in item_ids:
            scores = self.get_implicit_explanation(user_id, [target_item])
            cf_indices = np.where(scores.cpu().numpy() > 0.5)[0]
            cf_results.extend(cf_indices.tolist())
            
        return self.convert_cf_list_to_mask(torch.tensor(cf_results, device=self.device, dtype=torch.long))

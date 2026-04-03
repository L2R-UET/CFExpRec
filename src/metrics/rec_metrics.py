import torch

def recall_at_k(pred, gt, k):
    pred = pred[:, :k]  # (n_users, k)
    mask_gt = (gt != -1).float() 

    hits = (pred.unsqueeze(2) == gt.unsqueeze(1)).float()  
    hits_per_user = (hits.sum(dim=1))   # (n_users, max_len)

    # Normalize by number of positive items for each user
    recall = (hits_per_user.sum(dim=1) / mask_gt.sum(dim=1))   # (n_users,)
    return recall

def ndcg_at_k(pred, gt, k):

    pred = pred[:, :k]  # (n_users, k)
    mask_gt = (gt != -1).float() 
    device = pred.device
    positions = torch.arange(1, pred.size(1) + 1, device=device)  # (k,)
    dcg_weights = 1.0 / torch.log2(positions + 1.0)   # (K,)
    dcg_weights = dcg_weights.view(1, k, 1)
    
    hits = (pred.unsqueeze(2) == gt.unsqueeze(1)).float()
    dcg = (hits * dcg_weights).sum(dim=1).sum(dim=1)  # (n_users,)
    ideal_len = mask_gt.sum(dim=1).long()  # number of relevant per user

    idcg = torch.zeros_like(ideal_len, dtype=torch.float32, device=device)
    for i in range(1, k+1):
        idcg += (i <= ideal_len).float() * (1.0 / torch.log2(torch.tensor(i+1., device=device)))
    
    ndcg = dcg / idcg.clamp(min=1e-9)
    return ndcg
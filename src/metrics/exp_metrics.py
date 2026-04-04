import numpy as np
import torch

def pn_s_item_one_instance(list_before, list_after, item_pos, k=5):
    item_before_k = list_before[item_pos]
    list_after_k = list_after[:k]
    return int(item_before_k not in list_after_k)

def pn_s_list_one_instance(list_before, list_after, k=5):
    list_before_k = list_before[:k]
    list_after_k = list_after[:k]
    return len(set(list_after_k) - set(list_before_k)) / k

def pn_r_one_instance(list_before, list_after, k=5):
    list_before_k = list_before[:k]
    list_after_k = list_after[:k]
    items_change = set(list_after_k) - set(list_before_k)
    matching_indices = np.array([index + 1 for index, value in enumerate(list_after) if value in items_change])
    denominator = (1./np.log2(np.arange(2, k+2))).sum()
    numerator = (1./np.log2((matching_indices + 1))).sum()
    return (numerator / denominator).item()

def exp_size_one_instance(exp_mask_output):
    return exp_mask_output.sum().item()

def pos_p_one_instance(rec_model, ori_output_list, importance_score, user_idx, item_position=None, T=10):
    sp = importance_score.to_sparse()
    indices = sp.indices()
    values = sp.values()
    
    sorted_idx = torch.argsort(values, descending=True)
    
    values_sorted = values[sorted_idx]
    indices_sorted = indices[:, sorted_idx]

    nnz = values_sorted.numel()
    base = nnz // T
    remainder = nnz % T
    
    chunk_sizes = torch.full((T,), base, dtype=torch.long, device=values.device)
    chunk_sizes[:remainder] += 1
    
    cumulative_sizes = torch.cumsum(chunk_sizes, dim=0)

    user_imp_log = []

    for size in cumulative_sizes:
        active_pos_indices = indices_sorted[:, :size]

        active_values = torch.ones(size, device=values.device)
        
        pos_mask_t = torch.sparse_coo_tensor(
            active_pos_indices,
            active_values,
            sp.shape,
            device=values.device
        ).to_dense()

        _, output_pos = rec_model.predict(user_idx, len(ori_output_list), 1 - pos_mask_t)
        output_pos = output_pos.tolist()

        if item_position != None:
            user_imp_log.append(int(ori_output_list[item_position] in output_pos))
        else:
            user_imp_log.append(1.-len(set(output_pos) - set(ori_output_list))/len(ori_output_list))

    return np.mean(user_imp_log)


def neg_p_one_instance(rec_model, ori_output_list, importance_score, user_idx, item_position=None, T=10):
    sp = importance_score.to_sparse()
    indices = sp.indices()
    values = sp.values()
    
    sorted_idx = torch.argsort(values, descending=True)
    
    values_sorted = values[sorted_idx]
    indices_sorted = indices[:, sorted_idx]

    nnz = values_sorted.numel()
    base = nnz // T
    remainder = nnz % T
    
    chunk_sizes = torch.full((T,), base, dtype=torch.long, device=values.device)
    chunk_sizes[:remainder] += 1
    
    cumulative_sizes = torch.cumsum(chunk_sizes, dim=0)

    user_imp_log = []

    for size in cumulative_sizes:
        active_neg_indices = indices_sorted[:, -size:]

        active_values = torch.ones(size, device=values.device)

        neg_mask_t = torch.sparse_coo_tensor(
            active_neg_indices,
            active_values,
            sp.shape,
            device=values.device
        ).to_dense()

        _, output_neg = rec_model.predict(user_idx, len(ori_output_list), 1 - neg_mask_t)
        output_neg = output_neg.tolist()

        if item_position != None:
            user_imp_log.append(int(ori_output_list[item_position] in output_neg))
        else:
            user_imp_log.append(1.-len(set(output_neg) - set(ori_output_list))/len(ori_output_list))

    return np.mean(user_imp_log)


def gini_one_instance(importance_scores):
    sp = importance_scores.to_sparse()
    m_s = sp.values().cpu().flatten()
    if len(m_s) == 0:
        raise ValueError("The importance scores are all zero, Gini coefficient is undefined.")
    if torch.max(m_s) != torch.min(m_s):
        m_s = (m_s - torch.min(m_s)) / (torch.max(m_s) - torch.min(m_s))
    else:
        return 0.0
    m_s_sorted, _ = torch.sort(m_s)
    n = len(m_s_sorted)
    m_s_sum = torch.sum(m_s_sorted)
    gini_sum = 0.0
    for k_idx in range(n):
        gini_sum += (m_s_sorted[k_idx].item() / m_s_sum.item()) * ((n - k_idx - 0.5) / n)
    return 1 - 2 * gini_sum
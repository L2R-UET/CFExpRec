import torch
import torch.nn as nn
from collections import deque
import time
from functools import wraps

def measure_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = func(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        if isinstance(result, list):
            elapsed = [elapsed / len(result) for _ in result]
        return result, elapsed
    return wrapper

class ExpBaseModel(nn.Module):
    def __init__(self, model, device, args, config):
        super(ExpBaseModel, self).__init__()
        self.rec_model = model
        for p in self.rec_model.parameters():
            p.requires_grad = False
        self._get_recommendation(args.top_k)
        self.device = device
        self.args = args
        self.config = config
        self.mode = None # explicit, implicit, or hybrid
        self.ui_mat = model.ui_mat
        if self.ui_mat.is_sparse:
            self.ui_mat = self.ui_mat.to_dense()
        self.ui_mat.requires_grad = False
    
    def _get_recommendation(self, topk):
        relevance_score, y_pred_indices = self.rec_model.predict(topk=topk)
        self.y_pred_scores = relevance_score
        self.y_pred_indices = y_pred_indices.detach()

    def get_historical_interactions(self, user_id, **kwargs):
        raise NotImplementedError("Please inherit the class for user vector or graph-based explanations!")

    def get_implicit_explanation(self, user_id, item_ids, **kwargs):
        raise NotImplementedError("The model does not support implicit explanations.")
    
    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        raise NotImplementedError("The model does not support explicit explanations.")
    
    @measure_time
    def explain(self, user_id, mode="implicit", level="list", **kwargs):
        if mode == "implicit":
            if level == "list":
                return self.get_implicit_explanation(user_id, self.y_pred_indices[user_id].tolist(), **kwargs)
            elif level == "item":
                return [self.get_implicit_explanation(user_id, item_id.item(), **kwargs) for item_id in self.y_pred_indices[user_id]]
            else:                
                raise ValueError("Unsupported explanation level: {}".format(level))
        elif mode == "explicit":
            if level == "list":
                return self.get_explicit_explanation(user_id, self.y_pred_indices[user_id].tolist(), **kwargs)
            elif level == "item":
                return [self.get_explicit_explanation(user_id, item_id.item(), **kwargs) for item_id in self.y_pred_indices[user_id]]
            else:
                raise ValueError("Unsupported explanation level: {}".format(level))
        else:
            raise ValueError("Unsupported explanation mode: {}".format(mode))

    def convert_cf_list_to_mask(self, cf_list):
        pass

    def flip_mask(self, cf):
        return torch.abs(1 - cf)

class UserVectorExpBaseModel(ExpBaseModel):
    def __init__(self, model, device, args, config):
        super(UserVectorExpBaseModel, self).__init__(model, device, args, config)

    def get_historical_interactions(self, user_id, item_ids=None):
        return self.ui_mat.index_select(0, user_id).to_dense().squeeze(0)
    
    def convert_cf_list_to_mask(self, cf_list):
        '''
        Input:
        - cf_list: list of counterfactual items
        Output: multi-hot vector of counterfactual items
        '''
        mask = torch.zeros(self.rec_model.n_items, device=cf_list.device)
        mask[cf_list] = 1.
        return mask
    
class GraphExpBaseModel(ExpBaseModel):
    def __init__(self, model, device, args, config):
        super(GraphExpBaseModel, self).__init__(model, device, args, config)
        self._user_to_items = {row['user_id']: row['item_ids'] 
                              for _, row in self.rec_model.data_handler.train_group_user.iterrows()}
        self._item_to_users = {row['item_id']: row['user_ids'] 
                              for _, row in self.rec_model.data_handler.train_group_item.iterrows()}

    def get_historical_interactions(self, user_id, item_ids, graph_perturb="khop"):
        if graph_perturb == "full":
            return self.ui_mat
        elif graph_perturb == "khop":
            return self._get_subgraph_khop(user_id, item_ids)
        elif graph_perturb == "indirect":
            return self._get_subgraph_indirect_link(user_id, item_ids)
        elif graph_perturb == "user_only":
            return self._get_user_vector(user_id)
        else:
            raise ValueError("Unsupported graph perturb: {}".format(graph_perturb))

    def _get_subgraph_indirect_link(self, target_user_id, target_item_ids):
        user_to_items = self._user_to_items
        item_to_users = self._item_to_users
        if isinstance(target_item_ids, int):
            target_item_ids = [target_item_ids]
        item_ids = set(target_item_ids)
        edges = set()
        
        first_hop_user_side = set(user_to_items[target_user_id])
        
        first_hop_item_side = set()
        user_intermediate = set()
        for item_id in item_ids:
            first_hop_item_side.update(item_to_users[item_id])
            
        for item in first_hop_user_side.copy():
            user_connected = set(item_to_users[item]) & first_hop_item_side
            if len(user_connected) == 0:
                first_hop_user_side.remove(item)
            else:
                edges.update([(user, item) for user in user_connected])
                user_intermediate.update(user_connected)
    
        edges.update([(target_user_id, item) for item in first_hop_user_side])
        for item_id in item_ids:
            user_connected = user_intermediate & set(item_to_users[item_id])
            edges.update([(user, item_id) for user in user_connected])

        subgraph_mat = torch.zeros_like(self.ui_mat, device=self.device)

        if edges:
            edge_index = torch.tensor(list(edges), device=self.device, dtype=torch.long)
            subgraph_mat[edge_index[:, 0], edge_index[:, 1]] = 1.0

        return subgraph_mat

    def _get_subgraph_khop(self, target_user_id, target_item_ids, khop=3):
        user_to_items = self._user_to_items
        item_to_users = self._item_to_users

        visited_users = set([target_user_id])
        if isinstance(target_item_ids, int):
            target_item_ids = [target_item_ids]
        visited_items = set(target_item_ids)

        queue = deque()

        queue.append(("user", target_user_id, 0))

        for item_id in target_item_ids:
            queue.append(("item", item_id, 0))

        subgraph_edges = set()
        
        while queue:
            node_type, node_id, depth = queue.popleft()
            
            if depth >= khop:
                continue

            if node_type == "user":
                neighbors = user_to_items.get(node_id, [])
                for item_id in neighbors:
                    subgraph_edges.add((node_id, item_id))
                    
                    if item_id not in visited_items:
                        visited_items.add(item_id)
                        queue.append(("item", item_id, depth + 1))

            else:
                neighbors = item_to_users.get(node_id, [])
                for user_id in neighbors:
                    subgraph_edges.add((user_id, node_id))
                    
                    if user_id not in visited_users:
                        visited_users.add(user_id)
                        queue.append(("user", user_id, depth + 1))

        subgraph_mat = torch.zeros_like(self.ui_mat, device=self.device)

        if subgraph_edges:
            edge_index = torch.tensor(list(subgraph_edges), device=self.device, dtype=torch.long)
            subgraph_mat[edge_index[:, 0], edge_index[:, 1]] = 1.0
        
        return subgraph_mat

    def _get_user_vector(self, target_user_id):
        edges = [(target_user_id, item) for item in self._user_to_items[target_user_id]]
        subgraph_mat = torch.zeros_like(self.ui_mat, device=self.device)

        if edges:
            edge_index = torch.tensor(list(edges), device=self.device, dtype=torch.long)
            subgraph_mat[edge_index[:, 0], edge_index[:, 1]] = 1.0
        
        return subgraph_mat
    
    def convert_cf_list_to_mask(self, cf_list):
        '''
        Input:
        - cf_list: (2, len(cf))
        Output: 0/1 mask
        '''
        mask = torch.zeros_like(self.ui_mat, device=cf_list.device)
        mask[cf_list[0], cf_list[1]] = 1.
        return mask

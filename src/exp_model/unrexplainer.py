'''
Based on the original implementation of paper "UNR-Explainer: Counterfactual Explanations for Unsupervised Node Representation Learning Models"
Original repository: https://github.com/hjkng/unrexplainer
'''

import numpy as np
import math
import networkx as nx
import torch
from exp_model.base_model import GraphExpBaseModel

class UNRExplainer(GraphExpBaseModel):
    def __init__(self, model, device, args, config):
        super(UNRExplainer, self).__init__(model, device, args, config)

        self.mode = "explicit"
        self.c1 = getattr(config, "c1", 1.)
        self.restart = getattr(config, "restart", 0.2)
        self.max_iter = getattr(config, "max_iter", 500)
        self.top_k = args.top_k

        self.expansion_num = int(2 * self.ui_mat.sum() / (self.ui_mat.size(0) + self.ui_mat.size(1)))
        self.bf_dist_rank = self._emb_dist_rank(args.top_k)

    def _emb_dist_rank(self, neighbors_cnt):
        _, bf_dist_rank = self.rec_model.predict(topk=neighbors_cnt)
        return bf_dist_rank
    
    def convert_adj_mat_to_nx_graph(self, adj_mat):
        adj_mat_sparse = adj_mat.to_sparse()
        indices = adj_mat_sparse.indices()

        G = nx.Graph()

        # Add edges
        for u, i in indices.t().tolist():
            user_node = f"u_{u}"
            item_node = f"i_{i}"
            G.add_edge(user_node, item_node)
            G.add_edge(item_node, user_node)

        # return G, adj_mat.sum()
        return G
    
    def UCB1(self, vi, N, n, c1):
        if n > 0:
            return vi + c1*(math.sqrt(math.log(N)/n))
        else:
            return math.inf
        
    def backprop(self, mcts, rw_path, value):
    
        mcts = self.reset_agent(mcts)
        if value > mcts.V:
            mcts.V = value
            mcts.Vi = mcts.V
        mcts.N += 1

        
        for i in rw_path:
            if i == -1:
                mcts = self.reset_agent(mcts)
            else:
                mcts = mcts.C[i]
                if value > mcts.V:
                    mcts.V = value
                    mcts.Vi = mcts.V
                mcts.N += 1

    def reset_agent(self, mcts):
        while mcts.parent != None:
            mcts = mcts.parent
        return mcts

    def reset_subg(self, initial_nd):
        subgraph = nx.Graph()
        subgraph.add_node(initial_nd)
        return subgraph
        
    def select(self, mcts):
        
        N = mcts.N
        subgraph = self.reset_subg(mcts.state)
        rw_path = []

        while mcts.C != None: 
            if np.random.rand() < self.restart: 
                mcts = self.reset_agent(mcts)
                rw_path.append(-1)
            else: 
                children = mcts.C
                if len(children) == 0: 
                    mcts = self.reset_agent(mcts)
                    rw_path.append(-1)
                    if mcts.parent == None:
                        break
                else:
                    try:
                        if (rw_path[-1] == -1) and (len(mcts.C)>=2):
                            s = np.argmax([self.UCB1(children[i].Vi, N, children[i].N, self.c1) for i in children])
                            nlst = list(range(0, len(mcts.C)))
                            nlst.remove(s)
                            s = np.random.choice(nlst, 1)[0]    
                        else:
                            s = np.argmax([self.UCB1(children[i].Vi, N, children[i].N, self.c1) for i in children])
                            
                    except IndexError:
                        s = np.argmax([self.UCB1(children[i].Vi, N, children[i].N, self.c1) for i in children])
                    
                    subgraph.add_edge(mcts.state, mcts.C[s].state)
                    mcts = mcts.C[s]
                    rw_path.append(s)

        return mcts, subgraph, rw_path
    
    def simulate(self, subgraph, initial_nd):
        edges_to_perturb = list(subgraph.edges())
        value = self.importance(edges_to_perturb, initial_nd)
        return value

    def importance(self, edges_to_perturb, nd_idx):
        nd_idx = int(nd_idx[2:])
        bf_top_idx = self.bf_dist_rank[nd_idx].cpu().numpy()
        mask = self.ui_mat.clone()  
        for src, tgt in edges_to_perturb:
            if src.startswith("i"):
                src, tgt = tgt, src
            src = int(src[2:])
            tgt = int(tgt[2:])
            mask[src, tgt] = 0.0

        with torch.no_grad():
            self.rec_model.eval()
            _, af_top_idx = self.rec_model.predict(nd_idx, topk=self.top_k, mask=mask)
            af_top_idx = af_top_idx.cpu().numpy()
            importance = 1 - len(np.intersect1d(bf_top_idx, af_top_idx))/self.top_k

        return importance
    
    def get_explicit_explanation(self, user_id, item_ids, **kwargs):
        interaction = self.get_historical_interactions(user_id, item_ids, kwargs["graph_perturb"])
        self.G = self.convert_adj_mat_to_nx_graph(interaction)
        if self.G.number_of_nodes() == 0:
            return torch.zeros_like(self.ui_mat)

        initial_nd = f"u_{user_id}"
        
        mcts = MCTS(self.G, initial_nd, None, self.expansion_num)
        mcts.expansion(mcts)
        
        impt_vl = []; impt_sbg = []; num_nodes = []
        
        importance = 0; num_iter = 0; patience = 0; argmax_impt = 0
        n_nodes_khop= nx.ego_graph(self.G, initial_nd, radius=2).number_of_nodes()    
        
        while importance < 1.0: 

            mcts = self.reset_agent(mcts)
            mcts, subgraph, rw_path = self.select(mcts)

            # expansion condition
            if (mcts.C == None) and (mcts.N > 0):
                mcts.expansion(mcts)
                if len(mcts.C) > 0: 
                    subgraph.add_edge(mcts.state, mcts.C[0].state)
                    mcts = mcts.C[0]
                    rw_path.append(0)
                else:
                    if subgraph.number_of_nodes() ==1:
                        break
            else:
                pass

            importance = self.simulate(subgraph, initial_nd)

            n_nodes = subgraph.number_of_nodes()
            num_nodes.append(n_nodes)        
            self.backprop(mcts, rw_path, importance)

            impt_vl.append(importance)
            impt_sbg.append(subgraph)
            num_iter += 1

            if n_nodes ==1:
                break    
            elif n_nodes_khop==2:
                break
            elif (n_nodes_khop==3)and(num_iter > self.max_iter):
                break   

            if importance > argmax_impt:
                argmax_impt = importance
                patience = 0
            else:
                patience += 1

                if (patience > 10) and (num_iter > self.max_iter):
                    break
                else:
                    pass
                
        if n_nodes==1:
            mask_actual = torch.zeros_like(self.ui_mat)

        else:
            max_score = max(impt_vl)    
            max_lst = np.where(np.array(impt_vl) == max_score)[0]
            min_nodes = min([v for i,v in enumerate(num_nodes) if i in max_lst])
            fn_idx = [i for i,v in enumerate(num_nodes) if v ==min_nodes and i in max_lst][0]
            fn_sbg = impt_sbg[fn_idx]

            mask_actual = torch.zeros_like(self.ui_mat)
            for src, tgt in list(fn_sbg.edges()):
                if src.startswith("i"):
                    src, tgt = tgt, src
                src = int(src[2:])
                tgt = int(tgt[2:])
                mask_actual[src, tgt] = 1.0
            
        return mask_actual

class MCTS:
    def __init__(self, G, nd, parent, expansion_num):
        self.G = G
        self.state = nd
        self.V = 0  
        self.N = 0 
        self.Vi = 0
        self.parent = parent
        self.C = None
        self.expansion_num = expansion_num
        
    def expansion(self, parent):
        n_lst = [n for n in self.G.neighbors(self.state)]
        n_lst_idx = np.random.choice(len(n_lst), min(self.expansion_num, len(n_lst)), replace=False)
        n_lst = [n_lst[idx] for idx in n_lst_idx]            
        self.C = {i: MCTS(self.G, v, parent, self.expansion_num) for i, v in enumerate(n_lst)}
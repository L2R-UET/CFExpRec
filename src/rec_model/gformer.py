'''
Based on the original implementation of paper "Graph Transformer for Recommendation"
Original repository: https://github.com/HKUDS/GFormer
'''

from .base_model import GraphRecBaseModel
import torch
import torch.nn as nn
import scipy.sparse as sp
from tqdm import tqdm
import numpy as np
import networkx as nx
import multiprocessing as mp
import random


class GFormer(GraphRecBaseModel):
    def __init__(self, data_handler, device, args, config, **kwargs):
        super(GFormer, self).__init__(data_handler, device, args, config, **kwargs)

        self.fixSteps = getattr(config, "fixSteps", 2)
        self.reg = getattr(config, "reg", 1e-5)
        self.ssl_reg = getattr(config, "ssl_reg", 1)
        self.gcn_layers = getattr(config, "gcn_layers", 2)
        self.sub = getattr(config, "sub", 0.1)
        self.keepRate = getattr(config, "keepRate", 0.9)
        self.ext = getattr(config, "ext", 0.5)
        self.reRate = getattr(config, "reRate", 0.8)
        self.addRate = getattr(config, "addRate", 0.01)
        self.ctra = getattr(config, "ctra", 1e-3)
        self.b2 = getattr(config, "b2", 1)
        self.pnn_layers = getattr(config, "pnn_layers", 1)
        self.anchor_set_num = getattr(config, "anchor_set_num", 32)
        self.latent_dim = getattr(config, "latent_dim", 64)
        self.head = getattr(config, "head", 4)
        self.gtw = getattr(config, "gtw", 0.1)
        self.uEmbeds = nn.Parameter(torch.empty(self.n_users, self.latent_dim))
        self.iEmbeds = nn.Parameter(torch.empty(self.n_items, self.latent_dim))
        self.initEmbeds()

        self.gcnLayers = nn.Sequential(*[GCNLayer() for _ in range(self.gcn_layers)])
        self.gtLayer = GTLayer(self.latent_dim, self.head)
        self.pnnLayers = nn.Sequential(*[PNNLayer(self.latent_dim, self.anchor_set_num, self.device) 
                                         for _ in range(self.pnn_layers)])
        self.masker = RandomMaskSubgraphs(self.n_users, self.n_items, self.sub, self.keepRate, self.ext, self.reRate, device)
        self.sampler = LocalGraph(self.gtLayer, self.n_users, self.n_items,
                                  self.latent_dim, self.anchor_set_num, self.addRate, self.device)
        self.ori_norm_adj_mat = self.get_A_tilde(self.ui_mat)

    def initEmbeds(self):
        nn.init.xavier_uniform_(self.uEmbeds)
        nn.init.xavier_uniform_(self.iEmbeds)

    def get_A_tilde(self, ui_mat, mask=None):
        if ui_mat.is_sparse:
            return self._build_norm_adj_sparse(ui_mat)
        else:
            return self._build_norm_adj_dense(ui_mat, mask=mask)
    
    def _build_norm_adj_dense(self, ui_mat, mask=None):
        if mask != None:
            ui_mat = self.apply_mask(ui_mat, mask)
        else:
            ui_mat = self.ui_mat

        zero_uu = torch.zeros((self.n_users, self.n_users), device=self.device)
        zero_ii = torch.zeros((self.n_items, self.n_items), device=self.device)
        
        # adjacency matrix
        A = torch.cat(
            [
                torch.cat([zero_uu, ui_mat], dim=1),
                torch.cat([ui_mat.t(), zero_ii], dim=1)
            ],
            dim=0
        )

        I = torch.eye(A.shape[0], device=self.device)

        A = A + I
        
        D = torch.diag(A.sum(dim=1)).detach()
        D_exp = D ** (-0.5)
        D_exp[torch.isinf(D_exp)] = 0.0
        A_tilde = D_exp @ A @ D_exp
        
        return A_tilde

    def _build_norm_adj_sparse(self, ui_mat):

        indices = ui_mat.indices()       
        values = ui_mat.values()          

        user_idx = indices[0]        
        item_idx = indices[1]       

        UR = torch.stack([
            user_idx,
            item_idx + self.n_users
        ], dim=0)

        BL = torch.stack([
            item_idx + self.n_users,
            user_idx
        ], dim=0)

        adj_indices = torch.cat([UR, BL], dim=1)
        adj_values = torch.cat([values, values], dim=0)

        adj_mat = torch.sparse_coo_tensor(
            adj_indices,
            adj_values,
            size=(self.n_users + self.n_items, self.n_users + self.n_items),
            device=self.device
        ).coalesce()

        identity_mat = torch.eye(adj_mat.shape[0], device=self.device).to_sparse()
        adj_mat = adj_mat + identity_mat

        rowsum = torch.sparse.sum(adj_mat, dim=1).to_dense()

        d_inv = torch.pow(rowsum + 1e-9, -0.5) # D^(-1/2)
        d_inv[torch.isinf(d_inv)] = 0.0

        diag_idx = torch.arange(len(d_inv), device=self.device)
        
        d_mat_inv = torch.sparse_coo_tensor(
            torch.vstack([diag_idx, diag_idx]),
            d_inv,
            size=(len(d_inv), len(d_inv)),
            device=self.device
        )

        norm_adj_mat = torch.sparse.mm(d_mat_inv, adj_mat)
        norm_adj_mat = torch.sparse.mm(norm_adj_mat, d_mat_inv)

        return norm_adj_mat

    def getEgoEmbeds(self):
        return torch.cat([self.uEmbeds, self.iEmbeds], axis=0)

    def propagate(self, sub, cmp, encoderAdj, decoderAdj=None, is_test=True, anchorset_id=None, dists_array=None):
        embeds = torch.cat([self.uEmbeds, self.iEmbeds], axis=0)
        embedsLst = [embeds]
        emb, _ = self.gtLayer(cmp, embeds)
        cList = [embeds, self.gtw*emb]
        emb, _ = self.gtLayer(sub, embeds)
        subList = [embeds, self.gtw*emb]

        for gcn in self.gcnLayers:
            embeds = gcn(encoderAdj, embedsLst[-1])
            embeds2 = gcn(sub, embedsLst[-1])
            embeds3 = gcn(cmp, embedsLst[-1])
            subList.append(embeds2)
            embedsLst.append(embeds)
            cList.append(embeds3)
        if is_test is False:
            for pnn in self.pnnLayers:
                embeds = pnn(embedsLst[-1], anchorset_id, dists_array)
                embedsLst.append(embeds)
        if decoderAdj is not None:
            embeds, _ = self.gtLayer(decoderAdj, embedsLst[-1])
            embedsLst.append(embeds)
        embeds = sum(embedsLst)
        cList = sum(cList)
        subList = sum(subList)

        if is_test:
            return embeds[:self.n_users], embeds[self.n_users:]
        else:
            return embeds[:self.n_users], embeds[self.n_users:], cList, subList
    
    def forward(self, users, pos_items, neg_items, sub, cmp, encoderAdj, decoderAdj, anchorset_id, dists_array):
        final_user_embed, final_item_embed, cList, subList = self.propagate(sub, cmp, encoderAdj, decoderAdj,
                                                                            anchorset_id=anchorset_id,
                                                                            dists_array=dists_array, is_test=False)
        ancEmbeds = final_user_embed[users]
        posEmbeds = final_item_embed[pos_items]
        negEmbeds = final_item_embed[neg_items]

        usrEmbeds2 = subList[:self.n_users]
        itmEmbeds2 = subList[self.n_users:]
        ancEmbeds2 = usrEmbeds2[users]
        posEmbeds2 = itmEmbeds2[pos_items]

        bprLoss = (-torch.sum(ancEmbeds * posEmbeds, dim=-1)).mean()
        scoreDiff = torch.sum(ancEmbeds2 * posEmbeds2, dim=-1) - torch.sum(ancEmbeds2 * negEmbeds, dim=-1)
        bprLoss2 = - (scoreDiff).sigmoid().log().sum() / len(users)

        regLoss = 0
        for W in self.parameters():
            regLoss += W.norm(2).square()
        regLoss = regLoss * self.reg

        contrastLoss = (self.contrast(users, final_user_embed) + self.contrast(pos_items, final_item_embed)) * self.ssl_reg + self.contrast(
                users,
                final_user_embed,
                final_item_embed) + self.ctra*self.contrastNCE(users, subList, cList)

        return bprLoss + regLoss + contrastLoss + self.b2*bprLoss2

    def contrast(self, nodes, allEmbeds, allEmbeds2=None):
        if allEmbeds2 is not None:
            pckEmbeds = allEmbeds[nodes]
            scores = torch.log(torch.exp(pckEmbeds @ allEmbeds2.T).sum(-1)).mean()
        else:
            uniqNodes = torch.unique(nodes)
            pckEmbeds = allEmbeds[uniqNodes]
            scores = torch.log(torch.exp(pckEmbeds @ allEmbeds.T).sum(-1)).mean()
        return scores

    def contrastNCE(self, nodes, allEmbeds, allEmbeds2):
        pckEmbeds = allEmbeds[nodes]
        pckEmbeds2 = allEmbeds2[nodes]
        scores = torch.log(torch.exp(pckEmbeds * pckEmbeds2).sum(-1)).mean()
        return scores

    def compute_scores(self, users, mask=None):
        if mask != None:
            norm_adj_mat = self.get_A_tilde(self.ui_mat, mask=mask)
        else:
            norm_adj_mat = self.ori_norm_adj_mat
        u_emb, i_emb = self.propagate(norm_adj_mat, norm_adj_mat, norm_adj_mat, is_test=True)
        if users != None:
            u_emb = u_emb[users]
        if isinstance(users, int) or (isinstance(users, list) and len(users) == 1):
            return (u_emb.unsqueeze(0) @ i_emb.T).squeeze()
        return u_emb @ i_emb.T
        
    def single_source_shortest_path_length_range(self, graph, node_range):
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff=None)
        return dists_dict

    def get_random_anchorset(self):
        num_nodes = self.n_users + self.n_items
        anchorset_id = np.random.choice(num_nodes, size=self.anchor_set_num, replace=False)
        graph = nx.Graph()
        graph.add_nodes_from(np.arange(num_nodes))

        allOneAdj = torch.sparse_coo_tensor(
            self.ori_norm_adj_mat.indices(),
            torch.ones_like(self.ori_norm_adj_mat.values()),
            size=self.ori_norm_adj_mat.shape,
            device=self.device
        ).coalesce()

        rows = allOneAdj._indices()[0, :]
        cols = allOneAdj._indices()[1, :]

        rows = np.array(rows.cpu())
        cols = np.array(cols.cpu())

        edge_pair = list(zip(rows, cols))
        graph.add_edges_from(edge_pair)
        dists_array = np.zeros((len(anchorset_id), num_nodes))

        dicts_dict = self.single_source_shortest_path_length_range(graph, anchorset_id)
        for i, node_i in enumerate(anchorset_id):
            shortest_dist = dicts_dict[node_i]
            for j, node_j in enumerate(graph.nodes()):
                dist = shortest_dist.get(node_j, -1)
                if dist != -1:
                    dists_array[i, j] = 1 / (dist + 1)
        
        return anchorset_id, dists_array

    def train_epoch(self, optimizer):

        final_loss_list = []
        self.data_handler.train_dataset.negative_sampling(self.n_items)
        anchorset_id, dists_array = self.get_random_anchorset()
        self.train()

        for i, batch in tqdm(enumerate(self.data_handler.train_loader)):
            if i % self.fixSteps == 0:
                att_edge, add_adj = self.sampler(self.ori_norm_adj_mat, self.getEgoEmbeds(), anchorset_id, dists_array)
                encoderAdj, decoderAdj, sub, cmp = self.masker(add_adj, att_edge)

            users, pos_items, neg_items = batch

            loss = self.forward(users, pos_items, neg_items, 
                                sub, cmp, encoderAdj, decoderAdj, 
                                anchorset_id, dists_array)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=20, norm_type=2)
            optimizer.step()

            final_loss_list.append(loss.item())

        return final_loss_list


class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        if isinstance(adj, torch.sparse.Tensor):
            return torch.spmm(adj, embeds)
        else:
            return torch.mm(adj, embeds)


class PNNLayer(nn.Module):
    def __init__(self, latent_dim, anchor_set_num, device):
        super(PNNLayer, self).__init__()

        self.latent_dim = latent_dim
        self.anchor_set_num = anchor_set_num
        self.device = device

        self.linear_out_position = nn.Linear(self.latent_dim, 1)
        self.linear_out = nn.Linear(self.latent_dim, self.latent_dim)
        self.linear_hidden = nn.Linear(2 * self.latent_dim, self.latent_dim)
        self.act = nn.ReLU()
        

    def forward(self, embeds, anchor_set_id, dists_array):
        torch.cuda.empty_cache()
        dists_array = torch.tensor(dists_array, dtype=torch.float32).to(self.device)
        set_ids_emb = embeds[anchor_set_id]
        set_ids_reshape = set_ids_emb.repeat(dists_array.shape[1], 1).reshape(-1, len(set_ids_emb),
                                                                              self.latent_dim)  # 69534.256.32
        dists_array_emb = dists_array.T.unsqueeze(2)  #
        messages = set_ids_reshape * dists_array_emb  # 69000*256*32

        self_feature = embeds.repeat(self.anchor_set_num, 1).reshape(-1, self.anchor_set_num, self.latent_dim)
        messages = torch.cat((messages, self_feature), dim=-1)
        messages = self.linear_hidden(messages).squeeze()

        outposition1 = torch.mean(messages, dim=1)

        return outposition1


class GTLayer(nn.Module):
    def __init__(self, latent_dim, head):
        super(GTLayer, self).__init__()
        self.latent_dim = latent_dim
        self.head = head
        self.qTrans = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
        self.kTrans = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
        self.vTrans = nn.Parameter(torch.empty(self.latent_dim, self.latent_dim))
        self.initEmbeds()

    def initEmbeds(self):
        nn.init.xavier_uniform_(self.qTrans)
        nn.init.xavier_uniform_(self.kTrans)
        nn.init.xavier_uniform_(self.vTrans)

    def makeNoise(self, scores):
        noise = torch.rand(scores.shape).cuda()
        noise = -torch.log(-torch.log(noise))
        return scores + 0.01*noise

    def forward(self, adj, embeds):
        """
        adj: dense (N x N) adjacency matrix
        embeds: (N x D)
        """
        if adj.is_sparse:
            indices = adj._indices()
            rows, cols = indices[0, :], indices[1, :]
            rowEmbeds = embeds[rows]
            colEmbeds = embeds[cols]

            qEmbeds = (rowEmbeds @ self.qTrans).view([-1, self.head, self.latent_dim // self.head])
            kEmbeds = (colEmbeds @ self.kTrans).view([-1, self.head, self.latent_dim // self.head])
            vEmbeds = (colEmbeds @ self.vTrans).view([-1, self.head, self.latent_dim // self.head])

            att = torch.einsum('ehd, ehd -> eh', qEmbeds, kEmbeds)
            att = torch.clamp(att, -10.0, 10.0)
            expAtt = torch.exp(att)
            tem = torch.zeros([adj.shape[0], self.head]).cuda()
            attNorm = (tem.index_add_(0, rows, expAtt))[rows]
            att = expAtt / (attNorm + 1e-8)

            resEmbeds = torch.einsum('eh, ehd -> ehd', att, vEmbeds).view([-1, self.latent_dim])
            tem = torch.zeros([adj.shape[0], self.latent_dim]).cuda()
            resEmbeds = tem.index_add_(0, rows, resEmbeds)  # nd
            return resEmbeds, att
        
        else: 
            N = embeds.shape[0]
            H = self.head
            D = self.latent_dim // H

            # Project embeddings
            Q = (embeds @ self.qTrans).view(N, H, D)  # (N, H, D)
            K = (embeds @ self.kTrans).view(N, H, D)
            V = (embeds @ self.vTrans).view(N, H, D)

            # Compute attention logits: (N, N, H)
            att_logits = torch.einsum("nhd,mhd->nmh", Q, K)

            att_logits = torch.clamp(att_logits, -10.0, 10.0)

            # Mask non-edges using adjacency
            mask = (adj > 0).unsqueeze(-1)  # (N, N, 1)
            att_logits = att_logits.masked_fill(~mask, float("-inf"))

            # Softmax over neighbors
            att = torch.softmax(att_logits, dim=1)  # normalize over cols

            # Aggregate values
            out = torch.einsum("nmh,mhd->nhd", att, V)
            out = out.reshape(N, self.latent_dim)

            return out, att            

class LocalGraph(nn.Module):

    def __init__(self, gtLayer, n_users, n_items, latent_dim, anchor_set_num, add_rate, device):
        super(LocalGraph, self).__init__()
        self.gt_layer = gtLayer
        self.sft = nn.Softmax(0)
        self.device = device
        self.num_users = n_users
        self.num_items = n_items
        self.add_rate = add_rate
        self.pnn = PNNLayer(latent_dim, anchor_set_num, device).cuda()

    def makeNoise(self, scores):
        noise = torch.rand(scores.shape).cuda()
        noise = -torch.log(-torch.log(noise))
        return scores + noise

    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def merge_dicts(self, dicts):
        result = {}
        for dictionary in dicts:
            result.update(dictionary)
        return result

    def single_source_shortest_path_length_range(self, graph, node_range, cutoff):
        dists_dict = {}
        for node in node_range:
            dists_dict[node] = nx.single_source_shortest_path_length(graph, node, cutoff)
        return dists_dict

    def all_pairs_shortest_path_length_parallel(self, graph, cutoff=None, num_workers=1):
        nodes = list(graph.nodes)
        random.shuffle(nodes)
        if len(nodes) < 50:
            num_workers = int(num_workers / 4)
        elif len(nodes) < 400:
            num_workers = int(num_workers / 2)
        num_workers = 1  # windows
        pool = mp.Pool(processes=num_workers)
        results = self.single_source_shortest_path_length_range(graph, nodes, cutoff)

        output = [p.get() for p in results]
        dists_dict = self.merge_dicts(output)
        pool.close()
        pool.join()
        return dists_dict

    def forward(self, adj, embeds, anchorset_id, dists_array):

        embeds = self.pnn(embeds, anchorset_id, dists_array)
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        tmp_rows = np.random.choice(rows.cpu(), size=[int(len(rows) * self.add_rate)])
        tmp_cols = np.random.choice(cols.cpu(), size=[int(len(cols) * self.add_rate)])

        add_cols = torch.tensor(tmp_cols).to(self.device)
        add_rows = torch.tensor(tmp_rows).to(self.device)

        newRows = torch.cat([add_rows, add_cols, torch.arange(self.num_users + self.num_items).cuda(), rows])
        newCols = torch.cat([add_cols, add_rows, torch.arange(self.num_users + self.num_items).cuda(), cols])

        ratings_keep = np.ones(newRows.numel())
        adj_mat = sp.csr_matrix((ratings_keep, (newRows.cpu(), newCols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        add_adj = self.sp_mat_to_sp_tensor(adj_mat).to(self.device)

        _, atten = self.gt_layer(add_adj, embeds)
        att_edge = torch.sum(atten, dim=-1)

        return att_edge, add_adj


class RandomMaskSubgraphs(nn.Module):
    def __init__(self, num_users, num_items, sub, keep_rate, ext, rerate, device):
        super(RandomMaskSubgraphs, self).__init__()
        self.flag = False
        self.num_users = num_users
        self.num_items = num_items
        self.sub = sub
        self.keep_rate = keep_rate
        self.ext = ext
        self.rerate = rerate
        self.device = device
        self.sft = nn.Softmax(1)

    def normalizeAdj(self, adj):
        degree = torch.pow(torch.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return torch.sparse.FloatTensor(adj._indices(), newVals, adj.shape)
    def sp_mat_to_sp_tensor(self, sp_mat):
        coo = sp_mat.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
        return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()

    def create_sub_adj(self, adj, att_edge, flag):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]
        if flag:
            att_edge = (np.array(att_edge.detach().cpu() + 0.001))
        else:
            att_f = att_edge
            att_f[att_f > 3] = 3
            att_edge = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))  # 基于mlp可以去除
        att_f = att_edge / att_edge.sum()
        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * self.sub),
                                      replace=False, p=att_f)

        keep_index.sort()

        drop_edges = []
        i = 0
        j = 0
        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), rows])
        cols = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), cols])

        ratings_keep = np.ones(rows.numel())
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)
        return encoderAdj

    def forward(self, adj, att_edge):
        users_up = adj._indices()[0, :]
        items_up = adj._indices()[1, :]

        att_f = att_edge
        att_f[att_f > 3] = 3
        att_f = 1.0 / (np.exp(np.array(att_f.detach().cpu() + 1E-8)))
        att_f1 = att_f / att_f.sum()

        keep_index = np.random.choice(np.arange(len(users_up.cpu())), int(len(users_up.cpu()) * self.keep_rate),
                                          replace=False, p=att_f1)
        keep_index.sort()
        rows = users_up[keep_index]
        cols = items_up[keep_index]
        rows = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), rows])
        cols = torch.cat([torch.arange(self.num_users + self.num_items).cuda(), cols])
        drop_edges = []
        i, j = 0, 0

        while i < len(users_up):
            if j == len(keep_index):
                drop_edges.append(True)
                i += 1
                continue
            if i == keep_index[j]:
                drop_edges.append(False)
                j += 1
            else:
                drop_edges.append(True)
            i += 1

        ratings_keep = np.ones(rows.numel())
        adj_mat = sp.csr_matrix((ratings_keep, (rows.cpu(), cols.cpu())),
                                shape=(self.num_users + self.num_items, self.num_users + self.num_items))

        rowsum = np.array(adj_mat.sum(1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)
        norm_adj_tmp = d_mat_inv.dot(adj_mat)
        adj_matrix = norm_adj_tmp.dot(d_mat_inv)
        encoderAdj = self.sp_mat_to_sp_tensor(adj_matrix).to(self.device)


        drop_row_ids = users_up[drop_edges]
        drop_col_ids = items_up[drop_edges]

        ext_rows = np.random.choice(rows.cpu(), size=[int(len(drop_row_ids) * self.ext)])
        ext_cols = np.random.choice(cols.cpu(), size=[int(len(drop_col_ids) * self.ext)])

        ext_cols = torch.tensor(ext_cols).to(self.device)
        ext_rows = torch.tensor(ext_rows).to(self.device)
        
        tmp_rows = torch.cat([ext_rows, drop_row_ids])
        tmp_cols = torch.cat([ext_cols, drop_col_ids])
        new_rows = np.random.choice(tmp_rows.cpu(), size=[int(adj._values().shape[0] * self.rerate)])
        new_cols = np.random.choice(tmp_cols.cpu(), size=[int(adj._values().shape[0] * self.rerate)])

        new_rows = torch.tensor(new_rows).to(self.device)
        new_cols = torch.tensor(new_cols).to(self.device)

        newRows = torch.cat([new_rows, new_cols, torch.arange(self.num_users + self.num_items).cuda(), rows])
        newCols = torch.cat([new_cols, new_rows, torch.arange(self.num_users + self.num_items).cuda(), cols])
        hashVal = newRows * (self.num_users + self.num_items) + newCols
        hashVal = torch.unique(hashVal)
        newCols = hashVal % (self.num_users + self.num_items)
        newRows = ((hashVal - newCols) / (self.num_users + self.num_items)).long()

        decoderAdj = torch.sparse_coo_tensor(
            indices=torch.stack([newRows, newCols], dim=0),
            values=torch.ones(newRows.size(0), device=newRows.device, dtype=torch.float32),
            size=adj.shape,
            device=newRows.device
        ).coalesce()

        sub = self.create_sub_adj(adj, att_edge, True)
        cmp = self.create_sub_adj(adj, att_edge, False)

        return encoderAdj, decoderAdj, sub, cmp
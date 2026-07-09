import torch
from torch import nn
from torch.nn import functional as F
import torch_geometric.nn as gnn


# GNN for edge embeddings
class EmbNet(nn.Module):
    def __init__(self, depth=12, feats=4, units=32, act_fn='silu', agg_fn='mean'):
        super().__init__()
        self.depth = depth
        self.feats = feats
        self.units = units
        self.act_fn = getattr(F, act_fn)
        self.agg_fn = getattr(gnn, f'global_{agg_fn}_pool')
        self.v_lin0 = nn.Linear(self.feats, self.units)
        self.v_lins1 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins2 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins3 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_lins4 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.v_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])
        self.e_lin0 = nn.Linear(1, self.units)
        self.e_lins0 = nn.ModuleList([nn.Linear(self.units, self.units) for i in range(self.depth)])
        self.e_bns = nn.ModuleList([gnn.BatchNorm(self.units) for i in range(self.depth)])

    def reset_parameters(self):
        raise NotImplementedError

    def forward(self, x, edge_index, edge_attr):
        x = x
        w = edge_attr
        x = self.v_lin0(x)
        x = self.act_fn(x)
        w = self.e_lin0(w)
        w = self.act_fn(w)
        for i in range(self.depth):
            x0 = x
            x1 = self.v_lins1[i](x0)
            x2 = self.v_lins2[i](x0)
            x3 = self.v_lins3[i](x0)
            x4 = self.v_lins4[i](x0)
            w0 = w
            w1 = self.e_lins0[i](w0)
            w2 = torch.sigmoid(w0)
            x = x0 + self.act_fn(self.v_bns[i](x1 + self.agg_fn(w2 * x2[edge_index[1]], edge_index[0])))
            w = w0 + self.act_fn(self.e_bns[i](w1 + x3[edge_index[0]] + x4[edge_index[1]]))
        return w


# general class for MLP
class MLP(nn.Module):
    @property
    def device(self):
        return self._dummy.device

    def __init__(self, units_list, act_fn):
        super().__init__()
        self._dummy = nn.Parameter(torch.empty(0), requires_grad = False)
        self.units_list = units_list
        self.depth = len(self.units_list) - 1
        self.act_fn = getattr(F, act_fn)
        self.lins = nn.ModuleList([nn.Linear(self.units_list[i], self.units_list[i + 1]) for i in range(self.depth)])

    def forward(self, x):
        for i in range(self.depth):
            x = self.lins[i](x)
            if i < self.depth - 1:
                x = self.act_fn(x)
            else:
                x = torch.sigmoid(x) # last layer
        return x


# MLP for predicting parameterization theta
class ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        super().__init__([self.units] * depth + [self.preds], act_fn)

    def forward(self, x):
        return super().forward(x).squeeze(dim = -1)

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding"""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix

class AR_ParNet(MLP):
    def __init__(self, depth=3, units=32, preds=1, act_fn='silu'):
        self.units = units
        self.preds = preds
        self.solution_embedding =None
        super().__init__([self.units] * depth + [self.preds], act_fn)
        self.linear = nn.Linear(units * 2 + 1, units)

    def forward(self, x):
        x = self.linear(x)
        return super().forward(x).squeeze(dim = -1)

    def get_embedding(self, pyg, embedding, last, current):
        """Turn phe/heu vector into matrix with zero padding"""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes, self.units), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = embedding
        return matrix[last, current], matrix[current]

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding"""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix

class Net(nn.Module):
    def __init__(self, gfn=False, Z_out_dim=1, dual_decoder = False):
        super().__init__()
        self.emb_net = EmbNet()
        self.par_net_heu = ParNet()
        self.par_net_sl = ParNet() if dual_decoder else None

        self.gfn = gfn
        self.Z_net = nn.Sequential(
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, Z_out_dim),
        ) if gfn else None

    def forward(self, pyg, tw_normalize=True, return_logZ=False, return_embedding = False):
        x, edge_index, edge_attr = pyg.x, pyg.edge_index, pyg.edge_attr
        if tw_normalize:
            # shape: [n_nodes, 4] (loctions, tw)
            tw_end_max = x[0,-1]
            x[:, 2:] = x[:, 2:] / tw_end_max # tw/tw_end_max
            assert (x[:, 2] < x[:, 3]).all(), "Wrong Instance with wrong TW!"
        emb = self.emb_net(x, edge_index, edge_attr)
        heu = self.par_net_heu(emb)
        if self.par_net_sl is not None:
            pred_mask = self.par_net_sl(emb)
            heu = [heu, pred_mask]

        if return_logZ:
            assert self.gfn and self.Z_net is not None
            logZ = self.Z_net(emb).mean(0)
            if return_embedding:
                return heu, logZ, emb
            return heu, logZ
        if return_embedding:
            return heu, emb
        return heu

    def freeze_gnn(self):
        for param in self.emb_net.parameters():
            param.requires_grad = False

    @staticmethod
    def reshape(pyg, vector):
        """Turn phe/heu vector into matrix with zero padding"""
        n_nodes = pyg.x.shape[0]
        device = pyg.x.device
        matrix = torch.zeros(size=(n_nodes, n_nodes), device=device)
        matrix[pyg.edge_index[0], pyg.edge_index[1]] = vector
        return matrix

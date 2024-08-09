from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn

from dgl.nn.pytorch import GraphConv
import torch_sparse
from torch_sparse import SparseTensor, matmul

import numpy as np


class HyperNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(HyperNetwork, self).__init__()
        # 定义超网络的结构
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # 通过超网络生成权重
        x = torch.relu(self.fc1(z))
        weights = self.fc2(x)
        return weights


def get_sen(sens, idx_sens_train):
    sens_zeros = torch.zeros_like(sens)
    # print(f'sens={sens}')
    sens_1 = sens
    sens_0 = (1 - sens)

    sens_1[idx_sens_train] = sens_1[idx_sens_train] / len(sens_1[idx_sens_train])
    sens_0[idx_sens_train] = sens_0[idx_sens_train] / len(sens_0[idx_sens_train])

    # print(f'sens_1={sens_1.shape}')

    sens_zeros[idx_sens_train] = sens_1[idx_sens_train] - sens_0[idx_sens_train]

    sen_mat = torch.unsqueeze(sens_zeros, dim=0)
    # print(f'sen_mat={sen_mat[0, 0:10]}')
    # print(f'sen_mat={sen_mat[0, 10:20]}')

    return sen_mat


def check_sen(edge_index, sen):
    nnz = edge_index.nnz()
    deg = torch.eye(edge_index.sizes()[0]).cuda()
    adj = edge_index.to_dense()
    lap = (sen.t() @ sen).to_dense()
    lap2 = deg - adj
    diff = torch.sum(torch.abs(lap2 - lap)) / nnz
    assert diff < 0.000001, f'error: {diff} need to make sure L=B^TB'


class HyperFMP(nn.Module):
    def __init__(self, input_size, size, num_classes, num_task,num_layer,
                 in_feats: int,
                 out_feats: int,
                 K: int,
                 lambda1: float = None,
                 lambda2: float = None,
                 L2: bool = True,
                 dropout: float = 0.,
                 cached: bool = False,
                 **kwargs):
        super(HyperFMP, self).__init__()
        ray_hidden_dim = 128
        n_tasks = 3
        hidden_dim = 128
        out_dim = 2
        self.n_tasks = n_tasks
        self.ray_mlp2 = nn.Sequential(
            nn.Linear(n_tasks, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self.ray_mlp = nn.Sequential(
            nn.Linear(num_task, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(ray_hidden_dim, ray_hidden_dim),
        )

        self.in_dim = size
        #self.dims = [size, size, size]#最好 层数越多，auc越差，公平性越好
        self.dims = [size]
        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            setattr(self, f"fc_{i}_weights", nn.Linear(ray_hidden_dim, prvs_dim * dim))
            setattr(self, f"fc_{i}_bias", nn.Linear(ray_hidden_dim, dim))
            prvs_dim = dim

    def forward(self, ray):
        out_dict = dict()
        features = self.ray_mlp(ray)
        #print("ray", ray)
        #print("raysize", ray.size())
        #print("featuressize", features.size())

        weights = self.ray_mlp2(ray)
        weights = F.softmax(weights,dim=0)

        prvs_dim = self.in_dim
        for i, dim in enumerate(self.dims):
            out_dict[f"fc_{i}_weights"] = self.__getattr__(f"fc_{i}_weights")(
                features
            ).reshape(dim, prvs_dim)
            out_dict[f"fc_{i}_bias"] = self.__getattr__(f"fc_{i}_bias")(
                features
            ).flatten()
            prvs_dim = dim

        return out_dict,weights


class FMP(GraphConv):
    r"""Fair message passing layer
    """
    _cached_sen = Optional[SparseTensor]

    def __init__(self, input_size, size, num_classes, num_layer,
                 in_feats: int,
                 out_feats: int,
                 K: int,
                 lambda1: float = None,
                 lambda2: float = None,
                 L2: bool = True,
                 dropout: float = 0.,
                 cached: bool = False,
                 **kwargs):

        super(FMP, self).__init__(in_feats, out_feats)
        self.hidden = nn.ModuleList()
        for _ in range(num_layer):
            self.hidden.append(nn.Linear(size, size))
        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)
        self.K = K
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.L2 = L2
        self.dropout = dropout
        self.cached = cached

        # 初始化超网络

        self._cached_sen = None  ## sensitive matrix

        self.propa = GraphConv(in_feats, in_feats, weight=False, bias=False, activation=None)

        self.score = nn.Parameter(torch.from_numpy(np.array([.5])))

    def reset_parameters(self):
        self._cached_sen = None

    def forward(self, x: Tensor,
                g,
                idx_sens_train,
                edge_weight: OptTensor = None,
                sens=None,
                weights=None,weights2=None) -> Tensor:

        x = F.relu(self.first(x))
        for i in range(int(len(weights) / 2)):
            x = F.linear(x, weights[f"fc_{i}_weights"], weights[f"fc_{i}_bias"])
            if i < int(len(weights) / 2) - 1:
                x = F.relu(x)
        for layer in self.hidden:
            x = F.relu(layer(x))
        x = self.last(x)

        if self.K <= 0: return x

        cache = self._cached_sen
        if cache is None:
            sen_mat = get_sen(sens=sens, idx_sens_train=idx_sens_train)  ## compute sensitive matrix

            if self.cached:
                self._cached_sen = sen_mat
                self.init_z = torch.zeros((sen_mat.size()[0], x.size()[-1])).cuda()
        else:
            sen_mat = self._cached_sen  # N,

        # print(x.shape)torch.Size([66569, 2])
        # exit()

        hh = x
        x = self.emp_forward(g, x=x, hh=hh, K=self.K, weights2=weights2)
        return x

    def emp_forward(self, g, x, hh, K, weights2):
        lambda1 = self.lambda1
        lambda2 = self.lambda2

        gamma = 1 / (1 + lambda2)
        beta = 1 / (2 * gamma)

        for _ in range(K):

            if lambda2 > 0:
                y = gamma * hh + (1 - gamma) * self.propa(g, feat=x)
            else:
                y = gamma * hh + (1 - gamma) * x  # y = x - gamma * (x - hh)

            if lambda1 > 0:
                x = y - gamma * F.softmax(y, dim=1)
                # x = x*self.score
                # x = x*0.7

            else:
                x = y  # z=0

            # if weights is not None:
            #     # weights = weights.reshape(2,2)
            #     # print(weights.shape)
            #     # exit()
            #     x=
            # else:
            #     x = self.hypernetwork(x)
            x = x * weights2
            x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def __repr__(self):
        return '{}(K={}, lambda1={}, lambda2={}, L2={})'.format(
            self.__class__.__name__, self.K, self.lambda1, self.lambda2, self.L2)
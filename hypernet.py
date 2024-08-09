import torch
import torch.nn.functional as F
from torch import nn
import torchvision.models as models
from dgl.nn.pytorch import GraphConv

class Hypernetwork(nn.Module):

    def __init__(
        self,
        n_tasks=3,
        hidden_dim=128,
        out_dim=2
        
    ):
        super().__init__()
        self.n_tasks = n_tasks


        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )




    def forward(self, ray):
        # weights = []
        weights = self.ray_mlp(ray)
        weights = F.softmax(weights,dim=0)
        return weights


class HyperFMP(nn.Module):

    def __init__(
            self,
            in_feats,
            n_tasks=3,
            hidden_dim=128,
            out_dim=2,

    ):
        super().__init__()
        self.n_tasks = n_tasks

        self.ray_mlp = nn.Sequential(
            nn.Linear(n_tasks, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, out_dim),
        )
        self.propa = GraphConv(in_feats, in_feats, weight=False, bias=False, activation=None)

    def forward(self, ray):
        # weights = []
        weights = self.ray_mlp(ray)
        weights = F.softmax(weights, dim=0)
        return weights

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear

from fmp import FMP
from model import HyperFMP
from model import FMP as HFMP
class FairGNN(torch.nn.Module):
    def __init__(self, prop, **kwargs):
        super(FairGNN, self).__init__()
        """
        self.hidden = nn.ModuleList()
        for _ in range(num_layer-2):
            self.hidden.append(nn.Linear(size, size))

        self.first = nn.Linear(input_size, size)
        self.last = nn.Linear(size, num_classes)
        """
        
        self.prop = prop

    def reset_parameters(self):
        """
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        """
        self.prop.reset_parameters()

    def forward(self, x, g, sens, idx_sens_train,weights,weights2):

        x = self.prop(x, sens=sens, g=g, idx_sens_train=idx_sens_train,weights=weights,weights2=weights2)
        # return F.log_softmax(x, dim=1)
        return x



def get_model(args, data):

    Model = FairGNN
    prop =  FMP(in_feats=data.num_features,
                out_feats=data.num_features,
                K=args.num_layers, 
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                L2=args.L2,
                cached=True,
                input_size = data.num_features,
                size = args.num_hidden,
                num_classes = data.num_classes,
                num_layer = args.num_gnn_layer,
                num_task=args.n_task,
    )
    model = Model(prop=prop).cuda()

    return model

def get_HFMP(args, data):

    Model = FairGNN
    prop =  HFMP(in_feats=data.num_features,
                out_feats=data.num_features,
                K=args.num_layers,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                L2=args.L2,
                cached=True,
                input_size = data.num_features,
                size = args.num_hidden,
                num_classes = data.num_classes,
                num_layer = args.num_gnn_layer,
                num_task=args.n_task,
    )
    model = Model(prop=prop).cuda()

    return model

def get_model_hyper(args, data):

    Model =  HyperFMP(in_feats=data.num_features,
                out_feats=data.num_features,
                K=args.num_layers,
                lambda1=args.lambda1,
                lambda2=args.lambda2,
                L2=args.L2,
                cached=True,
                input_size = data.num_features,
                size = args.num_hidden,
                num_classes = data.num_classes,
                num_layer = args.num_gnn_layer,
                num_task=args.n_task,
    )
    model = Model.cuda()

    return model
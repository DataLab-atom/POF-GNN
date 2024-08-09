import time
import matplotlib.pyplot as plt
import argparse
import numpy as np
import logging
import time
import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score
from sklearn.metrics import average_precision_score

from utils import load_data, accuracy,load_pokec, fair_metric, str2bool, BestResultTracker
from module import GAT, GCN, SGC, APPNP
from fairgnn import get_model,get_model_hyper,get_HFMP

from tqdm import tqdm

from hypernet import Hypernetwork
from metrics import fair_metric_gpu

from collections import defaultdict
from pathlib import Path
from tqdm import trange

from utils import (
    circle_points,
    count_parameters,
    get_device,
    save_args,
    set_logger,
    set_seed,
)
from phn import EPOSolver, LinearScalarizationSolver




# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0,
                    help='assigned gpu.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--prefix', type=str, default='FGNN')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')#nba=0.01
parser.add_argument('--weight_decay', type=float, default=1e-5,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=.5,
                    help='Dropout rate (1 - keep probability).')#nba=0.7
parser.add_argument('--dataset', type=str, default='pokec_z',
                    choices=['pokec_z','pokec_n', 'nba'])
parser.add_argument('--num-hidden', type=int, default=64,
                    help='Number of hidden units of classifier.')
parser.add_argument("--num-heads", type=int, default=4,
                        help="number of hidden attention heads")
parser.add_argument("--num-out-heads", type=int, default=1,
                    help="number of output attention heads")
parser.add_argument("--num-layers", type=int, default=5,
                    help="number of hidden layers")
parser.add_argument("--residual", action="store_true", default=False,
                    help="use residual connection")
parser.add_argument("--in-drop", type=float, default=.5,
                        help="input feature dropout")
parser.add_argument("--edge-drop", type=float, default=.5,
                    help="edge dropout")
parser.add_argument("--attn-drop", type=float, default=.5,
                    help="attention dropout")
parser.add_argument('--negative-slope', type=float, default=0.2,
                    help="the negative slope of leaky relu")
parser.add_argument("--bias", action='store_true', default=False,
            help="flag to use bias")
parser.add_argument('--acc', type=float, default=0.5,
                    help='the selected FairGNN accuracy on val would be at least this high')
parser.add_argument('--roc', type=float, default=0.5,
                    help='the selected FairGNN ROC score on val would be at least this high')
parser.add_argument('--running_times', type=int, default=1, help='number of running times')
parser.add_argument('--lambda1', type=float, default=3, help='fairness')
parser.add_argument('--lambda2', type=float, default=3, help='smoothness')
parser.add_argument('--num_gnn_layer', type=int, default=2, help='number of gnn layers')
parser.add_argument('--L2', type=str2bool, default=True)
parser.add_argument("--alpha", type=float, default=0.1, help="Teleport Probability")
# parser.add_argument('--sens_bn', type=bool, default=False, help='Binary sensitive attribute')
# solver_type
parser.add_argument(
    "--solver_type", type=str, choices=["ls", "epo"], default="epo", help="solver_type"
)
parser.add_argument("--n_rays", type=int, default=25, help="num. rays")
parser.add_argument("--n_task", type=int, default=3, help="num. task")

args = parser.parse_args()

RUNNING_TIME = args.running_times
# hyper = args.hyper

device = torch.device('cuda:{}'.format(args.gpu))
print(args)
#%%
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Load data
print(args.dataset)

if args.dataset != 'nba':
    if args.dataset == 'pokec_z':
        dataset = 'region_job'
    else:
        dataset = 'region_job_2'
    sens_attr = "region"
    predict_attr = "I_am_working_in_field" ##"I_am_working_in_field", "spoken_languages_indicator"

    seed = 20
    path="dataset/pokec/"
    test_idx=False
else:
    dataset = 'nba'
    sens_attr = "country"
    predict_attr = "SALARY"
    label_number = 100
    sens_number = 50
    seed = 20
    path = "dataset/NBA"
    test_idx = True
print(dataset)

def add_random_integers(tensor, n=1, m=75):
    n = int(tensor.shape[1] * 0.2 * 0.14)  
    # n = 1

    # vvv = torch.nonzero(tensor)
    # print(tensor.shape)
    # print(vvv.shape)
    # exit()
    # 0.14511956242310078 不为零的密度

    # 获取tensor的行数和列数
    rows, cols = tensor.shape
    
    # 遍历每一行
    for i in range(rows):
        # 生成n个随机位置
        random_positions = np.random.choice(cols, n, replace=False)
        
        # 生成n个不超过m的随机整数
        random_integers = np.random.randint(1, m+1, n)
        random_integers = torch.tensor(random_integers,dtype=torch.int32)
        
        # 在随机位置上加上随机整数
        tensor[i, random_positions] += random_integers
    
    return tensor



adj, features, labels, idx_train, idx_val, idx_test,sens,idx_sens_train = load_pokec(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    seed=seed,test_idx=test_idx)

"""
print("adj",adj)
print("features",features)
print("labels",labels)
print("idx_train",idx_train)
print("idx_val",idx_val)
print("idx_test",idx_test)
print("sens",sens)
print("idx_sens_train",idx_sens_train)
"""

# print(f'features={features.shape}')
# print(f'sens={sens}')
# exit()
# features = add_random_integers(features)
#print("features",features.size())
#%%
import dgl
from utils import feature_norm
# g = dgl.DGLGraph()
g = dgl.from_scipy(adj)

# g = dgl.DGLGraph()
# g.from_scipy_sparse_matrix(adj)
if dataset=="nba":
    features = feature_norm(features)

g = g.to(device)


# n_classes = torch.max(labels).item() + 1
n_classes = 2

# print(f'features={features.shape}')

labels[labels>1]=1

# print(f'test labels={labels}')
if sens_attr:
    sens[sens>0]=1

# create data
meta_data = parser.parse_args()
meta_data.num_features = features.shape[1]
meta_data.num_classes = n_classes

# add self loop
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)
n_edges = g.number_of_edges()
n_nodes = g.number_of_nodes()

# print(f'n_nodes={n_nodes}')
# print(f'n_edges={n_edges}')

# model = FairGNN(nfeat = features.shape[1], args = args)
# model.estimator.load_state_dict(torch.load("./checkpoint/GCN_sens_{}_ns_{}".format(dataset,sens_number)))


features = features.to(device)
labels = labels.to(device)
idx_train = idx_train.to(device)
idx_val = idx_val.to(device)
idx_test = idx_test.to(device)
sens = sens.to(device)

sens_train = sens
idx_sens_train = idx_sens_train.to(device)



performances = []
fairnesss = []
for run_time in range(RUNNING_TIME):
    
    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    log_path = f'log/{args.dataset}/{args.prefix}/num_layer={args.num_gnn_layer}/lambda2={args.lambda2}'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    fh = logging.FileHandler(log_path + f'/lambda1={args.lambda1}-{run_time}.log', mode='w')

    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)




    # Model and optimizer

    model = get_HFMP(args, meta_data)
    hnet = get_model_hyper(args, meta_data).to(device)
    model = model.to(device)

    """
    model = get_model(args, meta_data)
    model = model.to(device)
    hnet = Hypernetwork(
        n_tasks=args.n_task,
        hidden_dim=128,
        out_dim=2).to(device)
    """
    # ------
    # solver
    # ------
    solvers = dict(ls=LinearScalarizationSolver, epo=EPOSolver)

    solver_method = solvers[args.solver_type]
    if args.solver_type == "epo":
        solver = solver_method(n_tasks=args.n_task, n_params=count_parameters(hnet))
        #solver = solver_method(n_tasks=n_classes, n_params=count_parameters(hnet))
    else:
        # ls
        solver = solver_method(n_tasks=args.n_task)
        #solver = solver_method(n_tasks=n_classes)

    min_angle = 0.1
    max_angle = np.pi / 2 - 0.1
    test_rays = circle_points(args.n_rays, min_angle=min_angle, max_angle=max_angle)

    # Train model
    t_total = time.time()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    alpha=8

    tracker = BestResultTracker()
    cls_loss_list = []
    parity_loss_list = []
    equality_loss_list = []
    parity_equality_loss_list = []
    for epoch in range(args.epochs):
        t = time.time()
        ### inference
        # train_features = features[idx_train]
        model.train()
        train_labels = labels[idx_train]
        
        if alpha > 0:

            ray = torch.from_numpy(
                    np.random.dirichlet((alpha, )*args.n_task, 1).astype(np.float32).flatten()
            ).to(device)
            """
            ray = torch.from_numpy(
                np.random.dirichlet((alpha, )*n_classes, 1).astype(np.float32).flatten()
            ).to(device)
            """
        else:
            alpha = torch.empty(
                1,
            ).uniform_(0.0, 1.0)
            ray = torch.tensor([alpha.item(), 1 - alpha.item()]).to(device)

        # ray = torch.from_numpy(
        #     np.array([0.1,0.8,0.1]).astype(np.float32).flatten()
        # ).to(device)
        # ray = torch.from_numpy(
        #     np.array([0.1,0.8,0.9]).astype(np.float32).flatten()
        # ).to(device)
        #print("ray",ray)
        #print("ray", ray.size())
        weights,weights2 = hnet(ray)
        #print("features",features.size())([66569, 265])
        #print("labels", labels.size())([66569])
        all_logit = model(features, g, sens, idx_sens_train,weights,weights2)#([66569, 2])
        all_y = F.softmax(all_logit, dim=1)#([66569, 2])
        #print(all_logit)
        parity_loss, equality_loss = fair_metric_gpu(all_logit, labels, sens, idx_train)
        #parity_loss, equality_loss = fair_metric_gpu(all_y[:,1], labels, sens, idx_train)
        # print(f'train_labels={train_labels}')
        ### training loss
        cls_loss = criterion(all_logit[idx_train],train_labels.long())
        #print("all_logit[idx_train]",all_logit[idx_train])
        #print("train_labels",train_labels.long())
        alapha = 0.6
        beta = 1-alapha
        #print("alapha*cls_loss",alapha*cls_loss)

        cls_loss_list.append(cls_loss)
        parity_loss_list.append(parity_loss)
        equality_loss_list.append(equality_loss)
        parity_equality_loss_list.append(parity_loss+equality_loss)


        losses = torch.stack((alapha*cls_loss, beta*parity_loss, beta*equality_loss))

        print("losses",losses)
        print("ray", ray)
        # exit()

        ray = ray.squeeze(0)
        loss = solver(losses, ray, list(hnet.parameters()))

        optimizer.zero_grad()
        # cls_loss.backward()
        loss.backward()
        optimizer.step()

        model.eval()
        
        all_logit = model(features, g, sens, idx_sens_train,weights,weights2)
        all_y = F.softmax(all_logit, dim=1)
        # print(f'all_y={all_y}')
        # print(f'labels={labels}')
        acc_train = accuracy(all_y[idx_train, 1], labels[idx_train]).item()
        ap_train = average_precision_score(labels[idx_train].cpu().numpy(), all_y[idx_train, 1].detach().cpu().numpy())
        roc_train = roc_auc_score(labels[idx_train].cpu().numpy(),all_y[idx_train, 1].detach().cpu().numpy())

        true_labels_train = labels[idx_train].cpu().numpy()
        pred_probs_train = all_y[idx_train, 1].detach().cpu().numpy()
        threshold = 0.5
        pred_labels_train = (pred_probs_train >= threshold).astype(int)
        f1_train = f1_score(true_labels_train, pred_labels_train)


        parity_train, eo_train = fair_metric(all_y[:, 1], labels, sens, idx_train)

        acc_val = accuracy(all_y[idx_val, 1], labels[idx_val]).item()
        ap_val = average_precision_score(labels[idx_val].cpu().numpy(), all_y[idx_val, 1].detach().cpu().numpy())
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(),all_y[idx_val, 1].detach().cpu().numpy())

        
        parity_val, eo_val = fair_metric(all_y[:, 1], labels, sens, idx_val)

        acc_test = accuracy(all_y[idx_test, 1], labels[idx_test]).item()
        ap_test = average_precision_score(labels[idx_test].cpu().numpy(), all_y[idx_test, 1].detach().cpu().numpy())
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(),all_y[idx_test, 1].detach().cpu().numpy())

        true_labels = labels[idx_test].cpu().numpy()
        pred_probs = all_y[idx_test, 1].detach().cpu().numpy()
        threshold = 0.5
        pred_labels = (pred_probs >= threshold).astype(int)
        f1_test = f1_score(true_labels, pred_labels)

        parity, eo = fair_metric(all_y[:, 1], labels, sens, idx_test)

        tracker.update(acc_test, parity, eo)

        logger.info('epoch: {}:'.format(epoch))
        logger.info(f'train acc: {acc_train:.4f}, val acc: {acc_val:.4f}, test acc: {acc_test:.4f}')
        logger.info(f'train ap: {ap_train:.4f}, val ap: {ap_val:.4f}, test ap: {ap_test:.4f}')
        # logger.info(f'train f1: {train_f1}, test f1: {val_f1}')
        logger.info(f'train auc: {roc_train:.4f}, val auc: {roc_val:.4f}, test auc: {roc_test:.4f}')
        logger.info('D_SP: {:.4f}, val D_SP: {:.4f}, test D_SP: {:.4f}'\
                    .format(parity_train, parity_val, parity))
        logger.info('D_EO: {:.4f}, val D_EO: {:.4f}, test D_EO: {:.4f}'\
                    .format(eo_train, eo_val, eo))



    cls_loss_list = [t.cpu().detach().numpy() for t in cls_loss_list]
    parity_equality_loss_list = [t.cpu().detach().numpy()*5 for t in parity_equality_loss_list]

    """
    plt.figure(figsize=(8, 6))
    plt.scatter(cls_loss_list, parity_equality_loss_list, cmap='viridis')
    plt.colorbar(label='Weight for Loss Function 1')
    plt.xlabel('cls_loss_list')
    plt.ylabel('parity_equality_loss_list')
    plt.title('Simulated Pareto Front')
    plt.show()
    plt.savefig('pareto_front.png')
    """
    print("cls_loss_list",cls_loss_list[-50:])
    print("parity_equality_loss_list",parity_equality_loss_list[-50:])

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    print('============performace on test set=============')

    logger.info(f'test acc: {acc_test:.4f}, test ap: {ap_test:.4f}, test auc: {roc_test:.4f}')
    logger.info('test D_SP: {:.4f}, test D_EO: {:.4f}'.format(parity, eo))
    logger.info("Total time elapsed: {:.4f}s".format(time.time() - t_total))
    logger.info(f'train f1: {f1_train}, test f1: {f1_test}')
    ## record performance and fairness metrics
    performances.append([acc_test, roc_test, ap_test,f1_test])
    fairnesss.append([parity, eo])

    print(f'running time={time.time() - t_total}')
    if run_time < RUNNING_TIME - 1:
        fh.close()
        logger.removeHandler(fh)
    # 测试结束后获取最优结果
    best_results = tracker.get_best_results()
    # print("Best results after all tests:", best_results)

### statistical results
print(performances)
performance_mean = np.around(np.mean(performances, 0), 4)
performance_std = np.around(np.std(performances, 0), 4)
fairness_mean = np.around(np.mean(fairnesss, 0), 4)
fairness_std = np.around(np.std(fairnesss, 0), 4)

logger.info('Average of performance and fairness metric')
logger.info("Test statistics: -- acc: {:.4f}+-{:.4f}, auc: {:.4f}+-{:.4f}, ap: {:.4f}+-{:.4f}, f1: {:.4f}+-{:.4f}" \
            .format(performance_mean[0], performance_std[0], 
                    performance_mean[1], performance_std[1],
                    performance_mean[2], performance_std[2],
                    performance_mean[3], performance_std[3]))
logger.info('Test statistics: -- D_SP: {:.4f}+-{:.4f}, D_EO: {:.4f}+-{:.4f}'\
            .format(fairness_mean[0], fairness_std[0],\
            fairness_mean[1], fairness_std[1]))
fh.close()
logger.removeHandler(fh)
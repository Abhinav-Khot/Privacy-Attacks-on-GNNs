import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from models.gcn import GCN, embedding_GCN
from models.DPAR import DPAR
from topology_attack import PGDAttack
from utils import *
from dataset import Dataset
import argparse
from sklearn.metrics import roc_curve, auc, average_precision_score
import scipy.io as sio
import random
import os
import pickle
import scipy.sparse as sp

# def test(adj, features, labels, victim_model):
#     adj, features, labels = to_tensor(adj, features, labels, device=device)

#     victim_model.eval()
#     adj_norm = normalize_adj_tensor(adj)
#     output = victim_model(features, adj_norm)

#     loss_test = F.nll_loss(output[idx_test], labels[idx_test])
#     acc_test = accuracy(output[idx_test], labels[idx_test])
#     print("Test set results:", "loss= {:.4f}".format(loss_test.item()), "accuracy= {:.4f}".format(acc_test.item()))

    # return output.detach()

def dot_product_decode(Z):
    Z = F.normalize(Z, p=2, dim=1)
    Z = torch.matmul(Z, Z.t())
    adj = torch.relu(Z-torch.eye(Z.shape[0]))
    return adj

def preprocess_Adj(adj, feature_adj):
    n=len(adj)
    cnt=0
    adj=adj.numpy()
    feature_adj=feature_adj.numpy()
    for i in range(n):
        for j in range(n):
            if feature_adj[i][j]>0.14 and adj[i][j]==0.0:
                adj[i][j]=1.0
                cnt+=1
    print(cnt)
    return torch.FloatTensor(adj)

def transfer_state_dict(pretrained_dict, model_dict):
    state_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict.keys():
            state_dict[k] = v
        else:
            print("Missing key(s) in state_dict :{}".format(k))
    return state_dict

def metric(ori_adj, inference_adj, idx):
    real_edge = ori_adj[idx, :][:, idx].reshape(-1)
    pred_edge = inference_adj[idx, :][:, idx].reshape(-1)
    # real_edge = ori_adj.reshape(-1)
    # pred_edge = inference_adj.reshape(-1)
    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    index = np.where(real_edge == 0)[0]
    index_delete = np.random.choice(index, size=int(len(real_edge)-2*np.sum(real_edge)), replace=False)
    real_edge = np.delete(real_edge, index_delete)
    pred_edge = np.delete(pred_edge, index_delete)
    print("Inference attack AUC: %f AP: %f" % (auc(fpr, tpr), average_precision_score(real_edge, pred_edge)))
    #get the  number of edges predicted correctly
    #correct_edges = number of edges in which both real and predicted edge values are 1
    correct_edges = np.sum((real_edge == 1) & (pred_edge == 1))
    pcnt = correct_edges/np.sum(real_edge == 1)
    print("Number of edges predicted correctly:", correct_edges, pcnt)


def Auc(ori_adj, modified_adj, idx):
    real_edge = []
    pred_edge = []
    for i in idx:
        for j in idx:
            if i != j:
                real_edge.append(ori_adj[i][j])
                pred_edge.append(modified_adj[i][j])
                #pred_edge.append(np.dot(output[idx[i]], output[idx[j]])/(np.linalg.norm(output[idx[i]])*np.linalg.norm(output[idx[j]])))
                #pred_edge.append(-np.linalg.norm(output[idx[i]]-output[idx[j]]))
                #pred_edge.append(np.dot(features[idx[i]], features[idx[j]]) / (np.linalg.norm(features[idx[i]]) * np.linalg.norm(features[idx[j]])))

    fpr, tpr, threshold = roc_curve(real_edge, pred_edge)
    print(auc(fpr, tpr))
    return auc(fpr, tpr)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=15, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100,
                    help='Number of epochs to optimize in GraphMI attack.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default='cora',
                    choices=['cora', 'cora_ml', 'citeseer', 'polblogs', 'pubmed', 'AIDS', 'usair', 'brazil'], help='dataset')
parser.add_argument('--density', type=float, default=1.0, help='Edge density estimation')
parser.add_argument('--model', type=str, default='PGD', choices=['PGD', 'min-max'], help='model variant')
parser.add_argument('--nlabel', type=float, default=0.1)
parser.add_argument('--model_param_file', type=str, help='model parameter file')
parser.add_argument('--train_adj_feature_file', type=str, help='train adj and feature file')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
device = torch.device("cpu") #comment this later after fixing tensor device issues
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device != 'cpu':
    torch.cuda.manual_seed(args.seed)

# data = Dataset(root='', name=args.dataset, setting='GCN')
# adj, features, labels, init_adj = data.adj, data.features, data.labels, data.init_adj

# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
#choose the target nodes

b = np.load(args.train_adj_feature_file, allow_pickle = True)

adj = torch.from_numpy(b["train_adj_matrix"])
features = torch.from_numpy(b["train_attr_matrix"])
labels = b["train_labels"]
# print(type(adj))
n = adj.shape[0]
result = np.zeros((n, n))
init_adj = sp.csr_matrix(result)

print('Done')

idx_attack = np.array(random.sample(range(adj.shape[0]), int(adj.shape[0]*args.nlabel)))
print(f"Number of indices randomly selected: {len(idx_attack)}")
# idx_attack = np.array(range(adj.shape[0]))
print(idx_attack)
num_edges = int(0.5 * args.density * adj.sum()/adj.shape[0]**2 * len(idx_attack)**2)

adj, features, labels = preprocess((adj), features, labels, preprocess_adj=False, onehot_feature=False)
# to tensor
print(len(labels))
feature_adj = dot_product_decode(features)
preprocess_adj = preprocess_Adj(adj, feature_adj)
init_adj = torch.FloatTensor(init_adj.todense())
# initial adj is set to zero matrix

loaded_params = np.load(args.model_param_file, allow_pickle=True)
print(loaded_params['W1:0'].shape)
print(loaded_params['W2:0'].shape)
victim_model = DPAR(loaded_params)

# Setup Victim Model

# victim_model = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1, nhid=16,
#                    dropout=0.5, weight_decay=5e-4, device=device)

# victim_model = victim_model.to(device)
# victim_model.fit(features, adj, labels, idx_train, idx_val)

# embedding = embedding_GCN(nfeat=features.shape[1], nhid=16, device=device)
# embedding.load_state_dict(transfer_state_dict(victim_model.state_dict(), embedding.state_dict()))

# Setup Attack Model

model = PGDAttack(model=victim_model, nnodes=adj.shape[0], loss_type='CE', device=device)

model = model.to(device)


def main():
    model.attack(features, init_adj, labels, idx_attack, num_edges, epochs=args.epochs)
    inference_adj = model.modified_adj.cpu()
    # print('=== testing GCN on original(clean) graph ===')
    # test(adj, features, labels, victim_model)
    print('=== calculating link inference AUC&AP ===')
    print(inference_adj.numpy().sum(axis = 0))
    print(torch.max(inference_adj), torch.min(inference_adj))
    metric(adj.numpy(), inference_adj.numpy(), idx_attack)

    #output = embedding(features.to(device), torch.zeros(adj.shape[0], adj.shape[0]).to(device))
    #adj1 = dot_product_decode(output.cpu())
    #metric(adj.numpy(), adj1.detach().numpy(), idx_attack)


if __name__ == '__main__':
    main()

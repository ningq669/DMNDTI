import dgl
from utils import *
from model import *
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl.nn.pytorch as dglnn
from sklearn.metrics import roc_auc_score, f1_score
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity as cos

warnings.filterwarnings("ignore")
seed = 47
args = setup(default_configure,seed)
s = 47
in_size = 512
hidden_size = 256
out_size = 128
dropout = 0.5
lr = 0.0001
weight_decay = 1e-10
epochs = 1000
cl_loss_co = 1
reg_loss_co = 0.0001
fold = 0
dir = "../modelSave"

args['device'] = "cuda:0" if torch.cuda.is_available() else "cpu"



dtidata, graph, num, all_meta_paths = load_dataset("../data/")


dti_label = torch.tensor(dtidata[:, 2:3]).to(args['device'])


#torch.Tensor().long()
#高斯初始化
num_protein = 2694
num_drug = 2634

hd = torch.randn((num_drug, in_size))
hp = torch.randn((num_protein, in_size))
features_d = hd.to(args['device'])
features_p = hp.to(args['device'])
node_feature = [features_d, features_p]
#得到边的关系
dti_cl = get_clGraph(dtidata, "dti").to(args['device'])

cl = dti_cl
data = dtidata
label = dti_label


def main(tr,te,seed):
        all_acc = []
        all_roc = []
        all_pr = []
        for  i in range(len(tr)):
            f = open(f"{i}foldtrain.txt","w",encoding="utf-8")
            train_index = tr[i]
            for train_index_one in train_index:
                f.write(f"{train_index_one}\n")
            test_index = te[i]
            f = open(f"{i}foldtest.txt","w",encoding="utf-8")
            for train_index_one in test_index:
                f.write(f"{train_index_one}\n")
            model = DMNDTI(
                num_drug=num_drug,
                num_protein=num_protein,
                all_meta_paths=all_meta_paths,
                in_size=[hd.shape[1], hp.shape[1]],
                hidden_size=[hidden_size, hidden_size],
                hidden_size1 = out_size,
                out_size=[out_size, out_size],
                dropout=dropout,
            ).to(args['device'])
            # model.load_state_dict(torch.load(f"{dir}/net{i}.pth"))
            optim = torch.optim.Adam(lr=lr, weight_decay=weight_decay, params=model.parameters())
            best_acc = 0
            best_pr = 0
            best_roc = 0
            for epoch in tqdm(range(epochs)):
                loss, train_acc, task1_roc, task1_pr,test_acc, test_roc, test_pr = train(model,optim,
                                                                                              train_index,
                                                                                              test_index,
                                                                                              epoch, i)

                if test_acc > best_acc:
                   best_acc = test_acc
                if test_pr > best_pr:
                    best_pr = test_pr
                if test_roc > best_roc:
                    best_roc = test_roc
                    # torch.save(obj=model.state_dict(), f=f"{dir}/net.pth")

            all_acc.append(best_acc)
            all_roc.append(best_roc)
            all_pr.append(best_pr)
            print(f"fold{i}  auroc is {best_roc:.4f} aupr is {best_pr:.4f} ")

        print(f"{sum(all_acc) / len(all_acc):.4f},  {sum(all_roc) / len(all_roc):.4f} ,{sum(all_pr) / len(all_pr):.4f}")

def train(model, optim,train_index,test_index, epoch,fold):
        model.train()
        d, p, out = model(graph, node_feature, train_index, data)
        train_acc = (out.argmax(dim=1) == label[train_index].reshape(-1)).sum(dtype=float) / len(train_index)

        task1_roc = get_roc(out, label[train_index])
        task1_pr = get_pr(out, label[train_index])

        loss = F.nll_loss(out, label[train_index].reshape(-1).long())
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f" {epoch} epoch loss  {loss:.4f} train is acc  {train_acc:.4f}, train roc is {task1_roc:.4f},train pr is{task1_pr}")
        te_acc, te_task1_roc1, te_task1_pr = main_test(model, d, p, test_index, epoch, fold)

        return loss.item(), train_acc, task1_roc, task1_pr, te_acc, te_task1_roc1, te_task1_pr


def main_test(model, d, p, test_index, epoch, fold):
    model.eval()
    # model(graph, node_feature, train_index, data)
    out = model(graph, node_feature, test_index, data, iftrain=False, d=d, p=p)

    acc1 = (out.argmax(dim=1) == label[test_index].reshape(-1)).sum(dtype=float) / len(test_index)

    task_roc = get_roc(out, label[test_index])

    task_pr = get_pr(out, label[test_index])

    return acc1, task_roc, task_pr


train_indeces, test_indeces = get_cross(dtidata)
main(train_indeces, test_indeces, seed)

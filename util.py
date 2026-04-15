import dgl
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.nn import functional as F
from tqdm import tqdm
from torch_geometric.data import  Batch

def common_loss(emb1, emb2):
    emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
    emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
    emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
    emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
    cov1 = torch.matmul(emb1, emb1.t())
    cov2 = torch.matmul(emb2, emb2.t())
    cost = torch.mean((cov1 - cov2) ** 2)
    return cost
def plotROC(y, z, pstr=''):
    fpr, tpr, tt = metrics.roc_curve(y, z)
    roc_auc = roc_auc_score(y, z)
    plt.figure()
    plt.plot(fpr, tpr, 'o-')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.grid()
    plt.title('ROC ' + pstr + ' AUC: '+str(roc_auc_score(y, z)))


def evaluate_metrics(y, y_pred, y_proba, draw_roc=False):

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    
    ba = balanced_accuracy_score(y, y_pred)
    tpr = recall_score(y, y_pred)
    tnr = tn/(tn+fp)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    
    if draw_roc:
        plotROC(y, y_pred)
    
    return tn, fp, fn, tp, round(ba, 3), round(tpr, 3), round(tnr, 3), round(f1, 3), round(auc, 3)


def print_metrics(y_proba, y_actual):
    
    # arr_len = len(y_proba_arr)
    thresholds = np.linspace(0.01, 0.99, 99)
    best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc = 0, 0, 0, 0, 0, 0

    for threshold in thresholds:
        total_ba, total_tpr, total_tnr, total_f1, total_mcc, total_auc = 0, 0, 0, 0, 0, 0

        y_pred_list = (np.array(y_proba) >= threshold).astype(int)
        tn, fp, fn, tp, ba, tpr, tnr, f1, auc = evaluate_metrics(y_actual, y_pred_list, y_proba)
        mcc = ((tp * tn) - (fp * fn)) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        total_ba += ba
        total_tpr += tpr
        total_tnr += tnr
        total_f1 += f1
        total_mcc += mcc
        total_auc += auc

        if total_ba > best_ba:
            best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc = total_ba, total_tpr, total_tnr, total_f1, total_mcc, total_auc
    return best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc



def predict(args, model, data_loader, criterion, optimizer, is_train, epoch, t):

    device = args['DEVICE']

    total_loss, correct = 0, 0
    output_total, y_total, y_pred_total = [], [], []

    for i, X_data, y_data in tqdm(data_loader, desc='{}_epoch_{}'.format(t,epoch),leave=True):
        if args['MODEL'] == 'acgcn-mmp':
            smiles1 = [x[0]['GRAPH_SMILES1'] for x in X_data]
            smiles2 = [x[0]['GRAPH_SMILES2'] for x in X_data]
            pyg1 = [x[0]['pyg1'] for x in X_data]
            pyg2 = [x[0]['pyg2'] for x in X_data]
            y_data = torch.from_numpy(np.array(y_data)).float()

            batch_smiles1 = dgl.batch(smiles1)
            batch_smiles2 = dgl.batch(smiles2)
            pyg1 = Batch.from_data_list(pyg1)
            pyg2 = Batch.from_data_list(pyg2)

            if torch.cuda.is_available():
                batch_smiles1 = batch_smiles1.to(device)
                batch_smiles2 = batch_smiles2.to(device)
                pyg1 = pyg1.to(device)
                pyg2 = pyg2.to(device)
                y_data = y_data.to(device)

            outputs = model(batch_smiles1, batch_smiles2, pyg1, pyg2, i, t)

        elif args['MODEL'] == 'acgcn-sub':
            core = [x[0]['GRAPH_CORE'] for x in X_data]
            sub1 = [x[0]['GRAPH_SUB1'] for x in X_data]
            sub2 = [x[0]['GRAPH_SUB2'] for x in X_data]
            b1 = [x[0]['B1'] for x in X_data]
            b2 = [x[0]['B2'] for x in X_data]
            sub1_pyg = [x[0]['sub1_pyg'] for x in X_data]
            sub2_pyg = [x[0]['sub2_pyg'] for x in X_data]
            y_data = torch.from_numpy(np.array(y_data)).float()

            batch_core = dgl.batch(core)
            batch_sub1 = dgl.batch(sub1)
            batch_sub2 = dgl.batch(sub2)
            b1 = Batch.from_data_list(b1)
            b2 = Batch.from_data_list(b2)
            sub1_pyg = Batch.from_data_list(sub1_pyg)
            sub2_pyg = Batch.from_data_list(sub2_pyg)

            if torch.cuda.is_available():
                batch_core = batch_core.to(device)
                batch_sub1 = batch_sub1.to(device)
                batch_sub2 = batch_sub2.to(device)
                b1 = b1.to(device)
                b2 = b2.to(device)
                sub1_pyg = sub1_pyg.to(device)
                sub2_pyg = sub2_pyg.to(device)
                y_data = y_data.to(device)

            outputs, s1, p1, s2, p2 = model(batch_core, batch_sub1, batch_sub2, b1, b2, sub1_pyg, sub2_pyg, i, t)
        if args['MODEL'] == 'acgcn-mmp':
            loss = criterion(outputs, y_data)

        elif args['MODEL'] == 'acgcn-sub':
            loss1 = criterion(outputs, y_data)
            loss2 = common_loss(s1, p1)
            loss3 = common_loss(s2, p2)
            loss = loss1 + args['beta'] * loss2 + args['beta'] * loss3

        output_total += outputs.tolist()
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        y_pred = (outputs >= 0.5).float()
        correct += (y_pred == y_data).float().sum()
        y_total += y_data.tolist()
        y_pred_total += [int(i) for i in y_pred]

    bal_acc = balanced_accuracy_score(y_total, y_pred_total)

    return model, loss, total_loss, bal_acc, output_total


def get_actual_label(data_loader):
    
    y_arr = []
    for i, X_data, y_data in data_loader:
        y_arr += y_data.tolist()
    
    return y_arr


class WeightedBCELoss(torch.nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.weights = weights
        self.eps = 1e-9

    def forward(self, output, target):
        if self.weights is not None:
            assert len(self.weights) == 2
            loss = self.weights[1] * (target * torch.log(output + self.eps)) + \
                self.weights[0] * ((1 - target) * torch.log(1 - output + self.eps))
        else:
            loss = target * torch.log(output + self.eps) + (1 - target) * torch.log(1 - output + self.eps)
            print(output, target)
            print(loss)
        return torch.neg(torch.mean(loss))

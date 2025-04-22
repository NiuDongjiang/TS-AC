import warnings

import torch

from args import Args
from utils.train import train
from utils.util import *
from utils.data_loader import *
from model.TS_mmp import TS_MMP
from model.TS_sub import TS_SUB
from sklearn.model_selection import train_test_split
import numpy as np

warnings.filterwarnings("ignore")


def main(args, random_seed):

    print(args)
    device = args['DEVICE']
    random_seed = random_seed

    ### Make dataset ###
    data = pd.read_csv('./data/' + args['TARGET_NAME'] + '_mmps.csv')
    data['label'] = data['label'].astype(int)
    label = data['label']  # 是一个数据序列，代表各个类别的标签。

    counter = label.value_counts()  # 统计每个类别标签的数量，并返回一个系列对象，其中索引为类别标签，值为该类别的计数。
    tot = counter.sum()  # 计算所有类别标签的总数。
    class_weight = [tot / (2 * i) for i in
                        counter]  # 列表推导式的目的是计算每个类别的权重，这通常用于处理数据集中类别不平衡的问题。权重计算方式是总数除以该类别数量的两倍。这样，较少的类别将获得更高的权重，而常见的类别获得较低的权重，从而在模型训练过程中平衡各类别的影响力。

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=random_seed, stratify=label)

    if args['MODEL'] == 'TS-mmp':
        model = TS_MMP(args).to(device)
        model.load_state_dict(torch.load('/tmp/TS-main/result/pre_train.pkl'))
        train_loader = TS_MMP_Dataset(args, train_data, True)
        test_loader = TS_MMP_Dataset(args, test_data, False)

    elif args['MODEL'] == 'TS-sub':
        model = TS_SUB(args).to(device)
        train_loader = TS_SUB_Dataset(args, train_data, True)
        test_loader = TS_SUB_Dataset(args, test_data, False)

    y_actual = get_actual_label(test_loader)
    y_proba = train(args, model, train_loader, test_loader, class_weight)

    best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc = print_metrics(y_proba, y_actual)
    return best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc




if __name__ == '__main__':
    BA, TPR, TNR, F1, MCC, AUC = [], [], [], [], [], []
    #random_seeds = [random.randint(0, 1000000) for _ in range(50)]
    #random_seeds = [438099, 110977, 851332, 496299, 662855]
    #random_seeds = [340657, 909747, 242201, 541398, 115708]
    random_seeds = [110977]
    for random_seed in random_seeds:
        args = Args().params

        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(random_seed)
        random.seed(random_seed)

        best_ba, best_tpr, best_tnr, best_f1, best_mcc, best_auc = main(args, random_seed)
        BA.append(best_ba)
        TPR.append(best_tpr)
        TNR.append(best_tnr)
        F1.append(best_f1)
        MCC.append(best_mcc)
        AUC.append(best_auc)

        print("==============" "{}".format(random_seed), "Performance =================\n",
              '* BA :', "{:.3f}".format(best_ba), '\n',
              '* TPR :', "{:.3f}".format(best_tpr), '\n',
              '* TNR :', "{:.3f}".format(best_tnr), '\n',
              '* F1-score :', "{:.3f}".format(best_f1), '\n',
              '* MCC :', "{:.3f}".format(best_mcc), '\n',
              '* AUC :', "{:.3f}".format(best_auc), '\n',
              '======================================================')
    print("============== Average Performance =================\n",
          '* BA :', "{:.3f}".format(np.mean(BA)), "+", "{:.3f}".format(np.std(BA, ddof=1)), '\n',
          '* TPR :', "{:.3f}".format(np.mean(TPR)), "+", "{:.3f}".format(np.std(TPR, ddof=1)), '\n',
          '* TNR :', "{:.3f}".format(np.mean(TNR)), "+", "{:.3f}".format(np.std(TNR, ddof=1)), '\n',
          '* F1-score :', "{:.3f}".format(np.mean(F1)), "+", "{:.3f}".format(np.std(F1, ddof=1)), '\n',
          '* MCC :', "{:.3f}".format(np.mean(MCC)), "+", "{:.3f}".format(np.std(MCC, ddof=1)), '\n',
          '* AUC :', "{:.3f}".format(np.mean(AUC)), "+", "{:.3f}".format(np.std(AUC, ddof=1)), '\n',
          '======================================================')
import argparse
import torch
import random

class Args(object):
    parser = argparse.ArgumentParser(description='Arguments for TS-AC')
    parser.add_argument('--model', default='acgcn-sub', type=str, help='[\'acgcn-mmp\', \'acgcn-sub\']')
    parser.add_argument('--target_name', default='thrombin', type=str, help='[\'thrombin\', \'mu_opioid_receptor\', \'melanocortin_receptor_4\']')
    parser.add_argument('--random_seed', type=int, default=random.randint(0, 1000000), help='Random seed for data split')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch Size')
    parser.add_argument('--early_stopping_patience', type=int, default=40, help='Early stopping patience')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--drop_out', type=float, default=0.2, help='Dropout probability')
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--tr_layer', default=2, type=int, help='transformer layer')#2
    parser.add_argument('--tr_head', default=4, type=int, help='transformer head')#mmp=4
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--beta', default=0.1, type=float, help='beta')#0.1

    parse = parser.parse_args()

    params = {
        "MODEL": parse.model,
        "TARGET_NAME": parse.target_name,
        "RANDOM_SEED": parse.random_seed,
        "BATCH_SIZE": parse.batch_size,
        "EARLY_STOPPING_PATIENCE": parse.early_stopping_patience,
        "WEIGHT_DECAY": parse.weight_decay,
        "DROP_OUT": parse.drop_out,
        "DEVICE": parse.device,
        "tr_layer": parse.tr_layer,
        "tr_head": parse.tr_head,
        'lr': parse.lr,
        'beta': parse.beta
    }


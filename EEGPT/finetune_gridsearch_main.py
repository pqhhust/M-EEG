import argparse
import random

import numpy as np
import torch
import itertools

# import wandb

from datasets.downstream import pearl_kfold_dataset as pearl_dataset
from datasets.downstream import uet175_dataset, hos_108_dataset, text_108_dataset
from finetune_trainer_kfold import Trainer
from downstream import model_for_pearls as model_for_pearl, model_for_uet175, model_for_108, model_for_text_108


def main():
    parser = argparse.ArgumentParser(description='Big model downstream')
    parser.add_argument('--project_name', type=str, default='EEGFoundationModel', help='project_name')
    parser.add_argument('--modality_mode', type=str, default='multi_attend', help='modality_mode (mono (only EEG), multi (EEG + other modalities))')
    parser.add_argument('--num_folds', type=int, default=-1, help='fold to test, -1 means all folds')
    parser.add_argument('--seed', type=int, default=3407, help='random seed (default: 0)')
    parser.add_argument('--cuda', type=int, default=0, help='cuda number (default: 1)')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training (default: 32)')
    parser.add_argument('--optimizer', type=str, default='AdamW', help='optimizer (AdamW, SGD)')
    parser.add_argument('--clip_value', type=float, default=1, help='clip_value')
    parser.add_argument('--meta_dim', type=int, default=36, help='meta data dimension')
    parser.add_argument('--classifier', type=str, default='all_patch_reps',
                        help='[all_patch_reps, all_patch_reps_twolayer, '
                             'all_patch_reps_onelayer, avgpooling_patch_reps]')
    # all_patch_reps: use all patch features with a three-layer classifier;
    # all_patch_reps_twolayer: use all patch features with a two-layer classifier;
    # all_patch_reps_onelayer: use all patch features with a one-layer classifier;
    # avgpooling_patch_reps: use average pooling for patch features;

    """############ Downstream dataset settings ############"""
    parser.add_argument('--downstream_dataset', type=str, default='epilepsy-text',
                        help='[FACED, SEED-V, PhysioNet-MI, SHU-MI, ISRUC, CHB-MIT, BCIC2020-3, Mumtaz2016, '
                             'SEED-VIG, MentalArithmetic, TUEV, TUAB, BCIC-IV-2a, PEARL, UET175, epilepsy-bbb, epilepsy-text]')
    parser.add_argument('--datasets_dir', type=str,
                        default='/mnt/disk1/aiotlab/namth/EEGFoundationModel/datasets/text_epilepsy_preprocessed',
                        help='datasets_dir')
    parser.add_argument('--num_of_classes', type=int, default=2, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='./data/wjq/models_weights/Big/BigFaced', help='model_dir')
    """############ Downstream dataset settings ############"""

    parser.add_argument('--num_workers', type=int, default=16, help='num_workers')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='label_smoothing')
    parser.add_argument('--multi_lr', type=bool, default=False,
                        help='multi_lr')  # set different learning rates for different modules
    parser.add_argument('--frozen', type=bool,
                        default=False, help='frozen')
    parser.add_argument('--use_pretrained_weights', type=bool,
                        default=True, help='use_pretrained_weights')
    parser.add_argument('--foundation_dir', type=str,
                        default='/mnt/disk1/aiotlab/namth/EEGFoundationModel/CBraMod/pretrained_weights/pretrained_weights.pth',
                        help='foundation_dir')

    params = parser.parse_args()
    
    group = 'Reproduce'

    kq_grid_search = {}
    search_space_lr = [3e-4, 5e-4, 1e-3, 2e-3]
    search_space_dropout = [0.1, 0.2, 0.3, 0.5]
    search_space_weight_decay = [1e-2, 2e-2, 5e-2]
    # search_space_lr = [2e-3]
    # search_space_dropout = [0.3]
    search_space = list(itertools.product(search_space_lr, search_space_dropout, search_space_weight_decay))
    
    dir = params.datasets_dir
    # torch.cuda.set_device(params.cuda)
    
    for lr, dropout, weight_decay in search_space:
        params.lr = lr
        params.dropout = dropout
        params.weight_decay = weight_decay
        params.epochs = int(dropout * 100)  # just to use different epochs for different settings
        print(params)
        acc, kappa, f1 = 0, 0, 0
        pr_auc, roc_auc = 0, 0
        res = []
        for i in range(5):
            if params.num_folds != -1 and i != params.num_folds:
                continue
            # run = wandb.init(project=params.project_name, 
            #         group=group,
            #         name=f'data_{params.downstream_dataset}_seed_{params.seed}',
            #         config=vars(params))
            setup_seed(params.seed)
            print(f'the downstream dataset is {params.downstream_dataset}, fold {i}')
            if params.downstream_dataset in ['PEARL', 'epilepsy-bbb', 'epilepsy-text']:
                params.datasets_dir = f'{dir}/fold_{i}'
                if params.downstream_dataset == 'PEARL':
                    print('Model for PEARL loaded')
                    load_dataset = pearl_dataset.LoadDataset(params)
                    model = model_for_pearl.Model(params).cuda()
                elif params.downstream_dataset == 'epilepsy-text':
                    print('Model for epilepsy-text loaded')
                    load_dataset = text_108_dataset.LoadDataset(params)
                    model = model_for_text_108.Model(params).cuda()
                else:  # epilepsy-bbb
                    print('Model for epilepsy-bbb loaded')
                    load_dataset = hos_108_dataset.LoadDataset(params)
                    model = model_for_108.Model(params).cuda()
                data_loader = load_dataset.get_data_loader()
                t = Trainer(params, data_loader, model)
                loader = load_dataset.get_data_loader()
                acc_, pr_auc_, roc_auc_ = t.train_for_binaryclass()
                acc += acc_
                pr_auc += pr_auc_
                roc_auc += roc_auc_
                res.append((acc_, pr_auc_, roc_auc_))
            elif params.downstream_dataset == 'UET175':
                load_dataset = uet175_dataset.LoadDataset(num_fold=i, params=params)
                data_loader = load_dataset.get_data_loader()
                model = model_for_uet175.Model(params).cuda()
                t = Trainer(params, data_loader, model)
                acc_, kappa_, f1_ = t.train_for_multiclass()
                acc += acc_
                kappa += kappa_
                f1 += f1_
                res.append((acc_, kappa_, f1_))
            # run.finish()

        if params.num_folds == -1:
            acc /= 5
            kappa /= 5
            f1 /= 5
            pr_auc /= 5
            roc_auc /= 5
        if params.downstream_dataset in ['UET175']:
            print(f'5-fold cross validation results: acc {acc}, kappa {kappa}, f1 {f1}')
            if params.num_folds == -1:
                for i in range(5):
                    print(f'fold {i}: acc {res[i][0]:.5f}, kappa {res[i][1]:.5f}, f1 {res[i][2]:.5f}')
        elif params.downstream_dataset in ['PEARL', 'epilepsy-bbb', 'epilepsy-text']:
            print(f'5-fold cross validation results: acc {acc}, pr_auc {pr_auc}, roc_auc {roc_auc}')
            if params.num_folds == -1:
                for i in range(5):
                    print(f'fold {i}: acc {res[i][0]:5f}, pr_auc {res[i][1]:5f}, roc_auc {res[i][2]:5f}')
        kq_grid_search[(lr, dropout, weight_decay)] = (acc, pr_auc, roc_auc)
        
    print('Grid Search Results:')
    for lr, dropout, weight_decay in kq_grid_search:
        acc, pr_auc, roc_auc = kq_grid_search[(lr, dropout, weight_decay)]
        print(f'LR: {lr:.5f}, Dropout: {dropout:.2f}, Weight Decay: {weight_decay:.2f} => Acc: {acc:.5f}, PR_AUC: {pr_auc:.5f}, ROC_AUC: {roc_auc:.5f}')
    print('Done!!!!!')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # wandb.login(key='ca0cbcfb28dcbf7cd1b54e0a6d40d8a482fc730f')
    print('Start training...')
    main()
    # wandb.finish()

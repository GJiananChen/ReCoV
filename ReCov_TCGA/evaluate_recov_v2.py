from __future__ import print_function

import time
import argparse
import pdb
import os
import math
import sys
from timeit import default_timer as timer
from pathlib import Path
ROOT_PATH = str(Path(__file__).resolve().parent.parent/"Patch-GCN")
sys.path.append(ROOT_PATH)

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

### Internal Imports
from datasets.dataset_survival import Generic_WSI_Survival_Dataset, Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl, load_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code

### PyTorch Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from multiprocessing import set_start_method

#40x experiments
#AMIL No exclusion: 0.51816 \pm 0.0452
#AMIL exclusion: 0.5311021865238733 ± 0.04218923401363137

#10x experiments
#TMIL No exclusion: 0.5928531546621999 ± 0.009621734998788599
#TMIL with amil index :  0.6035734226689001 ± 0.015052612214186138
#TMIL with no exclusion(30) : 0.649357900614182 ± 0.01620939022594018
#TMIL with exclusion(30): 0.6323841429369067 ± 0.014787667544162447
#AMIL No exclusion: 0.5459519821328865 ± 0.019378431978532066
#AMIL with exclusion: 0.5916247906197655 ± 0.02298764013611356
#AMIL with exclusion Test performance: 0.5855946398659966 ± 0.00860941832156177 (2nd try to check consistency)
# Test performance: 0.5728643216080401 ± 0.01391170206802676 (AMIL 30 epochs without exclusion)
# 0.6023450586264657 ± 0.028308593149417045 (AMIL 30 epochs)

#TMIL excluding all months<=3: Test performance: 0.6481295365717477 ± 0.011793690592994673
#TMIL no exclusion: 0.6472 ± 0.01635278569540982
#TMIL with auc + nll metric: 0.61

def load_df(path):
    with open("/home/ramanav/Projects/TCGA_MIL/ReCov_TCGA/train_cases.npy","rb") as file:
        train_cases = np.load(file,allow_pickle=True)
    with open("/home/ramanav/Projects/TCGA_MIL/ReCov_TCGA/test_cases.npy","rb") as file:
        test_cases = np.load(file,allow_pickle=True)
    with open("/home/ramanav/Projects/TCGA_MIL/ReCov_TCGA/train_cases_test.npy","rb") as file:
        train_cases_test = np.load(file,allow_pickle=True)
    with open("/home/ramanav/Projects/TCGA_MIL/ReCov_TCGA/test_cases_test.npy","rb") as file:
        test_cases_test = np.load(file,allow_pickle=True)
    with open(path,"rb") as file:
        memory_amil = np.load(file)
    df_cases = pd.read_csv("/home/ramanav/Projects/TCGA_MIL/Patch-GCN/datasets_csv/tcga_brca_all_clean.csv.zip")
    df_cases = df_cases[["case_id","survival_months","age","censorship"]].drop_duplicates()
    df_exp = df_cases.loc[df_cases["case_id"].isin(train_cases)].sort_values(by="case_id").reset_index(drop=True)
    df_exp["memory"] = memory_amil
    uncensor_list = list(df_exp.loc[df_exp["censorship"]==0].sort_values(by="memory")[:5].index)
    censor_list = list(df_exp.loc[df_exp["censorship"]==1].sort_values(by="memory")[:25].index)
    return uncensor_list + censor_list

def main(args, exclusion):
    #### Create Results Directory
    if not os.path.isdir(args.results_dir):
        os.mkdir(args.results_dir)

    if args.k_start == -1:
        start = 0
    else:
        start = args.k_start
    if args.k_end == -1:
        end = args.k
    else:
        end = args.k_end

    latest_val_cindex = []

    #Assuming same seed as main_recov
    patient_df = dataset.slide_data[["case_id","label"]].drop_duplicates(keep="first")
    case_list = patient_df["case_id"].values
    outcome_list = patient_df["label"].values
    X_train, X_test, y_train, y_test = train_test_split(case_list,outcome_list,train_size=0.8,shuffle=True, stratify=outcome_list,random_state=args.seed)
    ### Gets the Train + Val Dataset Loader.
    train_dataset, test_dataset = dataset.return_splits(from_id=True,  train_case_ids=X_train, val_case_ids=X_test)
    # train_df = train_dataset.slide_data[["case_id","label"]].drop_duplicates(keep="first")
    train_df = train_dataset.slide_data[["case_id","label","survival_months"]].drop_duplicates(keep="first")
    # train_df = train_df.loc[train_df["survival_months"]<=2]
    train_df = train_df.sort_values(by=["case_id"])
    train_cases_list = train_df["case_id"].values  
    
    if len(exclusion)!=0:
        noise_identified = [train_cases_list[ids] for ids in exclusion]
        #Remove only less months samples
        # noise_identified = train_df.loc[(train_df["case_id"].isin(noise_identified)) & (train_df["survival_months"]<=12),"case_id"].values      
        print(f"Excluding: {noise_identified}")
        noise_free_identified = list(set(train_cases_list) - set(noise_identified))
        noise_free_train_dataset, noise_dataset = train_dataset.return_splits(from_id=True, train_case_ids=np.array(noise_free_identified),val_case_ids=np.array(noise_identified))
    else:
        noise_free_train_dataset = train_dataset

    print('training: {}, validation: {}'.format(len(noise_free_train_dataset), len(test_dataset)))
    datasets = (noise_free_train_dataset, test_dataset)
    ### Start 5-runs
    folds = np.arange(args.k)
    for i in range(args.k):
        start = timer()
        seed_torch(args.seed + i)

        ### Run Train-Val on Survival Task.
        if args.task_type == 'survival':
            val_latest, cindex_latest = train(datasets, i, args)
            latest_val_cindex.append(cindex_latest)

        ### Write Results for Each Split to PKL
        end = timer()
        print('Run %d Time: %f seconds' % (i, end - start))

    ### Finish 5-Fold CV Evaluation.
    if args.task_type == 'survival':
        results_latest_df = pd.DataFrame({'folds': folds, 'val_cindex': latest_val_cindex})

    if len(folds) != args.k:
        save_name = 'summary_partial_{}_{}.csv'.format(start, end)
    else:
        save_name = 'summary.csv'

    results_latest_df.to_csv(os.path.join(args.results_dir, 'summary_latest.csv'))
    print(results_latest_df)
    print(u"Test performance: {} \u00B1 {}".format(np.mean(results_latest_df["val_cindex"].values),np.std(results_latest_df["val_cindex"].values)))

EXCLUSION = True
# EXCLUSION = False
### Training settings
parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
### Checkpoint + Misc. Pathing Parameters
# parser.add_argument('--data_root_dir', type=str, default='/scratch/localdata0/vramanathan/TCGA_MIL_Patches_Ctrans_1MPP/', help='data directory')
parser.add_argument('--data_root_dir', type=str, default='/localdisk3/ramanav/TCGA_processed/TCGA_MIL_Patches_Ctrans_1MPP/', help='data directory')
parser.add_argument('--seed',            type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
parser.add_argument('--k',               type=int, default=5, help='Number of folds (default: 5)')
parser.add_argument('--k_start',         type=int, default=-1, help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',           type=int, default=-1, help='End fold (Default: -1, first fold)')
# parser.add_argument('--results_dir',     type=str, default='/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov_eval', help='Results directory (Default: ./results)')
parser.add_argument('--results_dir',     type=str, default='/localdisk3/ramanav/Results/TCGA_Results/Recov_eval', help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv', help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_brca', help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca_100)')
parser.add_argument('--log_data',        action='store_true', default=False, help='Log data using tensorboard')
parser.add_argument('--overwrite',       action='store_true', default=True, help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--testing',         action='store_true', default=False, help='debugging tool')
parser.add_argument('--censor_time',      type=float, default=150.0, help='Censoring patients beyond certain time period')

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['deepset', 'amil', 'mifcn', 'dgc', 'patchgcn'], default='tmil', help='Type of model (Default: mcat)')
parser.add_argument('--mode',            type=str, choices=['path', 'cluster', 'graph'], default='path', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--num_gcn_layers',  type=int, default=4, help = '# of GCN layers to use.')
parser.add_argument('--edge_agg',        type=str, default='spatial', help="What edge relationship to use for aggregation.")
parser.add_argument('--resample',        type=float, default=0.00, help='Dropping out random patches.')
parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=30, help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',              type=float, default=2e-4, help='Learning rate (default: 0.0001)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
parser.add_argument('--label_frac',      type=float, default=1.0, help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg',             type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true', default=False, help='Enable early stopping')

args = parser.parse_args()
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Creates Experiment Code from argparse + Folder Name to Save Results
args = get_custom_exp_code(args)
args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'
print("Experiment Name:", args.exp_code)

### Sets Seed for reproducible experiments.
def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(args.seed)

encoding_size = 768
settings = {'num_splits': args.k, 
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs, 
            'results_dir': args.results_dir, 
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt}
print('\nLoad Dataset')

if 'survival' in args.task:
    args.n_classes = 4
    study = '_'.join(args.task.split('_')[:2])
    if study == 'tcga_kirc' or study == 'tcga_kirp':
        combined_study = 'tcga_kidney'
    elif study == 'tcga_luad' or study == 'tcga_lusc':
        combined_study = 'tcga_lung'
    else:
        combined_study = study
    study_dir = '%s_10x_features' % combined_study
    dataset = Generic_MIL_Survival_Dataset(csv_path = '%s/%s/%s_all_clean.csv.zip' % (ROOT_PATH,args.dataset_path, combined_study),
                                           mode = args.mode,
                                        #    data_dir= os.path.join(args.data_root_dir, study_dir),
                                            data_dir = args.data_root_dir,
                                           shuffle = False, 
                                           seed = args.seed, 
                                           print_info = True,
                                           patient_strat= False,
                                           n_bins=4,
                                           label_col = 'survival_months',
                                           ignore=[],)
                                        #    censor_time=args.censor_time)
else:
    raise NotImplementedError

if isinstance(dataset, Generic_MIL_Survival_Dataset):
    args.task_type = 'survival'
else:
    raise NotImplementedError

### Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

### Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
if EXCLUSION:
    string_add = "_excluded"
else:
    string_add = ""
args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, str(args.exp_code) + '_s{}'.format(args.seed) + time.strftime("_%d%b_%H_%M_%S", time.gmtime()) + string_add)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

### Sets the absolute path of split_dir
args.split_dir = os.path.join(ROOT_PATH,'splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

# MEMORY_PATH = "/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/memory_TCGA_cindex_1auc_0.5_1_nosplit.npy"
# MEMORY_PATH = "/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov/5foldcv/TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32_s1/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_1_nosplit.npy"
# MEMORY_PATH = "/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov/5foldcv/AMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_AMIL_nll_surv_a0.0_5foldcv_gc32_s1/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_1_nosplit.npy"
# MEMORY_PATH = "/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov/5foldcv/TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32_s1/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_1_nosplit.npy"
# MEMORY_PATH = "/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov/5foldcv/TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32_s1/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/home/vramanathan/scratch/amgrp/TCGA_Results/10xfeature_MIL/Recov/5foldcv/AMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32/tcga_brca_AMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32_s1/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_lr5e-04_5foldcv_gc32_s1_25Sep_17_36/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_26Sep_21_58/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_26Sep_21_59/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_29Sep_20_26/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_03Oct_21_25/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_1_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_03Oct_21_26/tcgabrca_recov_results/memory_TCGA_cindex_1.5auc_0.5_1_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_04Oct_23_23/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_04Oct_23_57/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_05Oct_21_28/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_06Oct_15_23/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_06Oct_15_25/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_08Oct_00_51/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_08Oct_00_52/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_08Oct_00_54/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
# MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_16Oct_17_53_02/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"
MEMORY_PATH = "/localdisk3/ramanav/Results/TCGA_Results/Recov/5foldcv/TransMIL_nll_surv_a0.0_5foldcv_gc32/tcga_brca_TransMIL_nll_surv_a0.0_5foldcv_gc32_s1_16Oct_18_04_48/tcgabrca_recov_results/memory_TCGA_cindex_1auc_0.5_0_nosplit.npy"

settings.update({'memory_path':MEMORY_PATH})
with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))        

if __name__ == "__main__":
    start = timer()
    TOP_K = 20
    if not EXCLUSION:
        exclusion = []
    else:
        with open(MEMORY_PATH,"rb") as file:
            memory = np.load(file)
        exclusion = np.argsort(memory)[:TOP_K]
        # exclusion = load_df(MEMORY_PATH)
        # exclusion = np.where(memory<=0.8)[0]
    # results = main(args,[])
    results = main(args,exclusion)
    end = timer()
    print("Memory path : {}".format(MEMORY_PATH))
    print("finished!")
    print("end script")
    print('Script Time: %f seconds' % (end - start))
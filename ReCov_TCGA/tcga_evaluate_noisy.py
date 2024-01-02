import yaml
import sys
import os
import random

from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from matplotlib import pyplot as plt
import seaborn as sns

from ReCov_TCGA.sample import sample_folds
from utils.trainfuncs import TrainEngine_MIL_folds
import trainer


def set_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def splits_tcgabraca(path, exclusion=[]):
    df = pd.read_csv(path)
    # test_cases = df.loc[df["train"]==0,"case_id"].tolist()
    # df_fil = df.loc[(df["is_female"]==1) & (df["train"]==1)]
    df_fil = df.loc[(df["is_female"]==1)]
    # df = df.loc[df["is_female"]==1]
    #Form a searchable dictionary
    #replace slide id with the feature vector
    columns_needed = ["case_id","survival_months","censorship"]
    df_fil = df_fil[columns_needed]
    #rename some columns
    df_fil.rename(columns={"survival_months":"durations","censorship":"vital_status"},inplace=True)
    df_fil["vital_status"] = 1 - df_fil["vital_status"]
    df_fil.sort_values(by="case_id")
    df_fil.drop_duplicates(keep="first",inplace=True)
    df_fil.reset_index(drop=True,inplace=True)
    cases = df_fil.to_dict()["case_id"]
    train_list = set(list(np.arange(len(cases)))) - set(exclusion)
    train_cases = [cases[ids] for ids in train_list]
    outcomes = [df_fil.loc[df_fil["case_id"]==case,"vital_status"].item() for case in train_cases]
    # reverse_cases = {v: k for k, v in cases.items()}
    # return train_cases, test_cases
    return train_cases, outcomes, cases

CONFIG_PATH = "/home/vramanathan/Projects/TCGA_MIL/configs/attn_mil_tcga_folds_noisy_test.yml"
DATAFRAME_PATH = "/home/vramanathan/Projects/TCGA_MIL/data/tcga_brca_all_clean.csv"
RESULT_DIR = Path("./results/tcga_sample")
# MEMORY_PATH = RESULT_DIR / "memory_TCGA_cindex_1auc_0.5_1.npy"
MEMORY_PATH = RESULT_DIR / "memory_TCGA_cindex_1auc_0.5_1_nosplit.npy"
N_RUNS = 5
N_FOLDS = 5
RANDOM_STATE = 1
LAMDA = 1
TOP_K = 30
TAU = 0.5
FILTER_DROP = True
NOISY_DROP = 0.4

if not (RESULT_DIR).is_dir():
    os.mkdir(RESULT_DIR)

########################################################################################################################################################

identified = []
with open(str(MEMORY_PATH),"rb") as file:
    memory = np.load(file)
# exclusion = list(np.argsort(memory)[:25])
exclusion = []
# train_cases, test_cases = splits_tcgabraca(DATAFRAME_PATH,exclusion)
# cases, y_cases, cases_dict = splits_tcgabraca(DATAFRAME_PATH, exclusion)
# aucs_last = []
# print('----------With Noise------------------')
# for seed in range(RANDOM_STATE, RANDOM_STATE+N_RUNS):
#     set_seed(seed)
#     aucs_last_fold = []
#     print(f"Run {seed}")
#     kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
#     fold_splits = list(kfold.split(cases,y_cases))
#     for fold, (train_split, test_split) in enumerate(fold_splits):
#         train_split = [cases_dict[ids] for ids in train_split]
#         test_split = [cases_dict[ids] for ids in test_split]
#         deep_trainer = TrainEngine_MIL_folds(config_pth=CONFIG_PATH, train_split=train_split, test_split=test_split, random_seed=seed)
#         deep_trainer.run()
#         auc, survstatus, survtime_all, riskval, patient_ids = deep_trainer.get_val_metrics
#         aucs_last_fold.append(auc)
#     aucs_last.append(np.mean(aucs_last_fold))
# print(aucs_last)
# print(u"Test performance: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))


# exclusion = list(np.argsort(memory)[:30])
exclusion = list(np.argsort(memory)[:20])
cases, y_cases, cases_dict = splits_tcgabraca(DATAFRAME_PATH, exclusion)
aucs_last = []
print('----------Without Noise------------------')
for seed in range(RANDOM_STATE, RANDOM_STATE+N_RUNS):
    set_seed(seed)
    aucs_last_fold = []
    print(f"Run {seed}")
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold_splits = list(kfold.split(cases,y_cases))
    for fold, (train_split, test_split) in enumerate(fold_splits):
        train_split = [cases_dict[ids] for ids in train_split]
        test_split = [cases_dict[ids] for ids in test_split]
        deep_trainer = TrainEngine_MIL_folds(config_pth=CONFIG_PATH, train_split=train_split, test_split=test_split, random_seed=seed)
        deep_trainer.run()
        auc, survstatus, survtime_all, riskval, patient_ids = deep_trainer.get_val_metrics
        aucs_last_fold.append(auc)
    aucs_last.append(np.mean(aucs_last_fold))
print(aucs_last)
print(u"Test performance: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))




# print('----------With Noise------------------')
# aucs_last = []
# for seed in range(RANDOM_STATE,RANDOM_STATE+N_RUNS):
#     set_seed(seed)
#     print(f"Run {seed}")
#     # print(f"Fold number: {fold}")
#     deep_trainer = TrainEngine_MIL_folds(config_pth=CONFIG_PATH, train_split=train_cases, test_split=test_cases, random_seed=seed)
#     deep_trainer.run()
#     auc, survstatus, survtime_all, riskval, patient_ids = deep_trainer.get_val_metrics
#     # print(f"Fold {fold} test performance: {auc}")
#     aucs_last.append(auc)
#     #Generate new set of folds based on weights
# print(aucs_last)
# print(u"Test performance: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))

# exclusion = list(np.argsort(memory)[:20])
# print(exclusion)
# # exclusion = []
# train_cases, test_cases = splits_tcgabraca(DATAFRAME_PATH,exclusion)

# print('--------Noise Removed-------------------')
# aucs_last = []
# for seed in range(RANDOM_STATE,RANDOM_STATE+N_RUNS):
#     set_seed(seed)
#     print(f"Run {seed}")
#     # print(f"Fold number: {fold}")
#     deep_trainer = TrainEngine_MIL_folds(config_pth=CONFIG_PATH, train_split=train_cases, test_split=test_cases, random_seed=seed)
#     deep_trainer.run()
#     auc, survstatus, survtime_all, riskval, patient_ids = deep_trainer.get_val_metrics
#     # print(f"Fold {fold} test performance: {auc}")
#     aucs_last.append(auc)
#     #Generate new set of folds based on weights
# print(aucs_last)
# print(u"Test performance: {} \u00B1 {}".format(np.mean(aucs_last),np.std(aucs_last)))
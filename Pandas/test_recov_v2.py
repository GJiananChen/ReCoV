#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Wed Jan 03 2024 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is copied from test_recov
# Modifications (date, what was modified):
#   1. Changed to new version of modelling where I use multiclass classification instead
# --------------------------------------------------------------------------------------------------------------------------
#

import random
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
import torchmetrics

from mil_models import TransMIL_peg
from pandas_dataloader import Pandas_Dataset
from train_mil import preprocess_data,train_full,val_one_epoch,collate

#hyperparameters settings
parser = argparse.ArgumentParser(description='Configurations for Gleason Grading in Pandas dataset')
#system settings
parser.add_argument('--seed',type=int,default=1)
# parser.add_argument('--data_root_dir', type=str, default='/localdisk3/ramanav/TCGA_processed/PANDAS_MIL_Patches_Ctrans_1MPP/', help='data directory')
parser.add_argument('--data_root_dir', type=str, default='/aippmdata/public/PANDAS/PANDAS_MIL_Patches_Selfpipeline_1MPP/', help='data directory')
parser.add_argument('--csv_path', type=str, default='/aippmdata/public/PANDAS')
parser.add_argument('--save_dir',type=str, default='/home/vramanathan/Projects/ReCoV/results/pandas')
# parser.add_argument('--save_dir',type=str, default='/localdisk3/ramanav/Results/ReCoV/results/PANDAS')

#model settings
parser.add_argument('--num_classes',type=int, default=6)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--lamda', type=float, default=0.0005, help="weight decay to use in adam optimizer")
parser.add_argument('--patience', type=int, default=10, help="number of epochs to wait in reducelronplateu lr scheduler")
parser.add_argument('--num_epochs', type=int, default=30)
#recov settings
parser.add_argument('--n_folds',type=int,default=5)
# parser.add_argument('--model_name',type=str,default="15Dec_09_26_56_20_2.5_s1_[1, 1, 0]")
# parser.add_argument('--model_name',type=str,default="20Dec_15_47_06_20_2.5_s1_[0, 0, 0, 1]_0")
# parser.add_argument('--model_name',type=str,default="20Dec_16_02_12_20_1.0_s1_[0, 0, 0, 1]_1")
# parser.add_argument('--model_name',type=str,default="02Jan_19_37_48_20_s1_[1, 0]_1")
# parser.add_argument('--model_name',type=str,default="07Jan_16_56_02_1_s1_[1, 0]_0_origsoln")
# parser.add_argument('--model_name',type=str,default="12Jan_14_58_46_20_s1_[1, 0]_1_changemem")
# parser.add_argument('--model_name',type=str,default="17Jan_09_22_35_20_s1_[1, 0]_1_origfoldsplit")
# parser.add_argument('--model_name',type=str,default="01Feb_13_28_48_20_s1_0.1_[1, 0]_1_origfoldsplit")
parser.add_argument('--model_name',type=str,default="01Feb_13_28_48_20_s1_0.5_[1, 0]_1_origfoldsplit")

parser.add_argument('--exclusion',action='store_true',default=True)
args = parser.parse_args()
print(args)

MODEL_NAME = args.model_name
ROOT_PATH = Path(args.save_dir)
EXCLUSION = args.exclusion
DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = time.strftime("%d%b_%H_%M_%S", time.gmtime())
EPCOH_NUM = 15

#model defination
# model = TransMIL_peg(n_classes=args.num_classes-1)
model = TransMIL_peg(n_classes=args.num_classes-1, dim=512)

#Prepare data
(X_train_clean,y_train_clean),(X_val_clean,y_val_clean),(X_test_clean,y_test_clean), _ = preprocess_data(args)
X_train_clean["isup_grade"] = y_train_clean
train_split = X_train_clean.reset_index(drop=True)
X_val_clean["isup_grade"] = y_val_clean
# val_split = X_val_clean.reset_index(drop=True)
X_test_clean["isup_grade"] = y_test_clean
# X_test_clean = X_test_clean.loc[X_test_clean["data_provider"]=="radboud"]
# karolinska
# X_test_clean = X_test_clean.loc[X_test_clean["data_provider"]=="karolinska"]
train_split = pd.concat((X_train_clean,X_val_clean))
# train_split = train_split.sort_values(by="image_id").reset_index(drop=True)
# train_split["int_id"] = train_split.index
test_split = X_test_clean.reset_index(drop=True)


metricfunc = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=args.num_classes,average=None),
                                        torchmetrics.F1Score(num_classes=args.num_classes,average='weighted'),
                                        ])

testmetricdict = metricfunc.clone(prefix='test_').to(DEVICE)

#load recov data
if EXCLUSION:
    # path = ROOT_PATH/f"memory__{MODEL_NAME}_v2.npy"
    # path = ROOT_PATH/f"memory__{MODEL_NAME}_19_v2.npy"
    path = ROOT_PATH/f"memory__{MODEL_NAME}_{EPCOH_NUM}_v2.npy"
    split = pd.read_csv(ROOT_PATH/f"splitsave__{MODEL_NAME}.csv")
    data = np.load(path)
    split["memory"] = data
    orig_length = len(train_split)
    # split = pd.read_csv(ROOT_PATH/"local_preds_final_1_efficientnet-b1.csv")
    # split = pd.read_csv("/home/ramanav/Projects/ReCoV/results/pandas/local_preds_final_1_efficientnet-b1.csv")
    # split["memory"] = 5 - np.abs(split["probs_raw"]-split["isup_grade"])

    # THRESHOLD = np.percentile(split["memory"],5)
    THRESHOLD = np.percentile(split["memory"],10)
    print("Excluding training samples")
    print(split.loc[split["memory"]<=THRESHOLD]["data_provider"].value_counts())
    print(split.loc[split["memory"]<=THRESHOLD]["isup_grade"].value_counts())
    index_list = split.loc[split["memory"]>THRESHOLD,"image_id"].tolist()
    #random exclusion
    np.random.seed(int(time.time()))
    index_list = list(np.random.permutation(split["image_id"].tolist())[:len(index_list)])
    temp = split["image_id"].tolist()
    random.shuffle(temp)
    index_list = temp[:len(index_list)]
    
    train_split = train_split.loc[train_split["image_id"].isin(index_list)].reset_index(drop=True)
    print(f"Excluded: {orig_length-len(train_split)}")
    print(index_list)
    # val_split = val_split.loc[val_split["image_id"].isin(index_list)].reset_index(drop=True)

trainset = Pandas_Dataset(train_split,args.data_root_dir)
# valset = Pandas_Dataset(val_split,args.data_root_dir)
testset = Pandas_Dataset(test_split,args.data_root_dir)
# _,_,_,_  = train_full((trainset,valset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_{MODEL_NAME}")
# _,_,_,_  = train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_{MODEL_NAME}_{EXCLUSION*1}")
# _,_,_,_  = train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_{MODEL_NAME}_{EXCLUSION*1}_officialsoln")
# _,_,_,_  = train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_{MODEL_NAME}_{EXCLUSION*1}_officialsoln_recov")
# _,_,_,_  = train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_{MODEL_NAME}_{EPCOH_NUM}_{EXCLUSION*1}_bigger_recov_origmemsplit")
_,_,_,_  = train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_randomexclusion")

# model = torch.load(Path(args.save_dir)/f"{timestamp}_{MODEL_NAME}.pt")
# # model = torch.load(Path(args.save_dir)/f"{'_22Dec_15_33_13'}_{MODEL_NAME}.pt")
# # model = torch.load(Path(args.save_dir)/f"{'_22Dec_15_41_11'}_{MODEL_NAME}.pt")
# lossfunc = nn.CrossEntropyLoss().to(DEVICE)
# state = torch.load("../results/pandas/03Jan_noexclusion_tmil_statedict.pt")
# model = TransMIL_peg(5)
# model.load_state_dict(state)
# model.eval()
# model.to(DEVICE)
# lossfunc = nn.BCEWithLogitsLoss().to(DEVICE)
# testloader = torch.utils.data.DataLoader(testset,batch_size=args.batch_size,shuffle=False,collate_fn=collate)
# test_metric,test_loss,pred_probs,true_labels,image_ids = val_one_epoch(model,testloader,lossfunc,testmetricdict,DEVICE,verbosity=True)


# print("Final test metric: {}".format(test_metric))
"""
Test on test set and save model for testing on kaggle held out test dataset
"""
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torchmetrics

from mil_models import TransMIL_peg
from pandas_dataloader import Pandas_Dataset
from train_mil import preprocess_data,train_full

#hyperparameters settings
parser = argparse.ArgumentParser(description='Configurations for Gleason Grading in Pandas dataset')
#system settings
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--data_root_dir', type=str, default='../data/PANDAS/PANDAS_MIL_Patches_Selfpipeline_1MPP/', help='data directory')
parser.add_argument('--csv_path', type=str, default='../data/PANDAS')
parser.add_argument('--save_dir',type=str, default='../results/PANDAS')

#model settings
parser.add_argument('--num_classes',type=int, default=6)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr',type=float,default=1e-4)
parser.add_argument('--lamda', type=float, default=0.0005, help="weight decay to use in adam optimizer")
parser.add_argument('--patience', type=int, default=10, help="number of epochs to wait in reducelronplateu lr scheduler")
parser.add_argument('--num_epochs', type=int, default=30)
#recov settings
parser.add_argument('--n_folds',type=int,default=5)

parser.add_argument('--model_name',type=str,default="25Feb_12_42_48_20_s1_1.0_[1, 0]_0.8_origfoldsplit")

parser.add_argument('--exclusion',action='store_true',default=True)
args = parser.parse_args()
print(args)

MODEL_NAME = args.model_name
ROOT_PATH = Path(args.save_dir)
EXCLUSION = args.exclusion
DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
timestamp = time.strftime("%d%b_%H_%M_%S", time.gmtime())
EPOCH_NUM = 12

#model defination
model = TransMIL_peg(n_classes=args.num_classes-1, dim=512)

#Prepare data
(X_train_clean,y_train_clean),(X_val_clean,y_val_clean),(X_test_clean,y_test_clean), _ = preprocess_data(args)
X_train_clean["isup_grade"] = y_train_clean
train_split = X_train_clean.reset_index(drop=True)
X_val_clean["isup_grade"] = y_val_clean
X_test_clean["isup_grade"] = y_test_clean
train_split = pd.concat((X_train_clean,X_val_clean))
test_split = X_test_clean.reset_index(drop=True)


metricfunc = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=args.num_classes,average=None),
                                        torchmetrics.F1Score(num_classes=args.num_classes,average='weighted'),
                                        ])

testmetricdict = metricfunc.clone(prefix='test_').to(DEVICE)

#load recov data
if EXCLUSION:
    path = ROOT_PATH/f"memory__{MODEL_NAME}_{EPOCH_NUM}_v2.npy"
    split = pd.read_csv(ROOT_PATH/f"splitsave__{MODEL_NAME}.csv")
    data = np.load(path)
    split["memory"] = data
    orig_length = len(train_split)

    THRESHOLD = np.percentile(split["memory"],10)
    print("Excluding training samples")
    print(split.loc[split["memory"]<=THRESHOLD]["data_provider"].value_counts())
    print(split.loc[split["memory"]<=THRESHOLD]["isup_grade"].value_counts())
    index_list = split.loc[split["memory"]>THRESHOLD,"image_id"].tolist()
    
    #random exclusion
    # np.random.seed(int(time.time()))
    # index_list = list(np.random.permutation(split["image_id"].tolist())[:len(index_list)])
    # temp = split["image_id"].tolist()
    # random.shuffle(temp)
    # index_list = temp[:len(index_list)]
    
    train_split = train_split.loc[train_split["image_id"].isin(index_list)].reset_index(drop=True)
    print(f"Excluded: {orig_length-len(train_split)}")
    print(index_list)

trainset = Pandas_Dataset(train_split,args.data_root_dir)
testset = Pandas_Dataset(test_split,args.data_root_dir)
_,_,_,_  = train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"{timestamp}_{MODEL_NAME}_{EPOCH_NUM}_{EXCLUSION*1}_fastrecov")
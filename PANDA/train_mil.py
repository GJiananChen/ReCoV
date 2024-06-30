"""
Script for training MIL model
"""

from pathlib import Path
import time

import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import torchmetrics
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from warmup_scheduler import GradualWarmupScheduler

#constants for cosine scheduler
WARMUP_FACTOR = 10 
WARMUP_EGO = 1

def train_one_epoch(model, trainloader, optimizer, lossfun, metricfun, DEVICE, verbosity=False):
    running_loss = 0.0
    model.train()
    predicted_labels = []
    true_labels = []
    for data in trainloader:
        x = data["image"]
        labels = data["label"].to(DEVICE)
        # initialize gradients to zero
        optimizer.zero_grad() 
        logits_store = []
        for i in range(len(labels)):
            # go over each image one by one and pool them into a single prediction
            # forward pass
            logits = model(x[i].to(DEVICE))
            logits_store.append(logits)
        logits = torch.cat(logits_store)
        #compute Loss with respect to target
        targets = torch.sum(labels,dim=1)
        logits = torch.cat(logits_store)
        loss = lossfun(logits, labels)
        # _, predicted = torch.max(logits, 1)
        pred_sig = torch.sigmoid(logits).detach()
        predicted = pred_sig.sum(dim=1).round()
        
        #Metric Calculation
        # total_dice+=MetricFun(outputs, labels)
        metrics_calc = metricfun(predicted.type(torch.int16),targets.type(torch.int16))
        # back propagate
        loss.backward()
        # do SGD step i.e., update parameters
        optimizer.step()
        # by default loss is averaged over all elements of batch
        running_loss += loss.data
        predicted_labels.extend(predicted.cpu())
        true_labels.extend(targets.cpu())
    running_loss = running_loss.cpu().numpy()
    metrics_calc = metricfun.compute()
    # print(metrics_calc)
    metricfun.reset()
    f1score = metrics_calc["train_F1Score"].cpu().numpy().item()
    accuracy = metrics_calc["train_Accuracy"].cpu().numpy()
    qwk = quadratic_kappa_coefficient(torch.tensor(predicted_labels).type(torch.int64),torch.tensor(true_labels).type(torch.int64))
    if verbosity:
        print("train accuracy: {}, train f1 score: {}, train QWK: {}".format(accuracy,f1score,qwk))
    return running_loss

def val_one_epoch(model,testloader,lossfun,metricfun,DEVICE,verbosity=False):
    running_loss = 0.0
    # evaluation mode which takes care of architectural disablings
    model.eval()
    pred_probs = []
    true_labels = []
    image_ids = []
    predicted_labels = []
    with torch.no_grad():
        for data in testloader:
            x = data["image"]
            labels = data["label"].to(DEVICE)
            #Convert to GPU
            logits_store = []
            for i in range(len(labels)):
                logits = model(x[i].to(DEVICE))
                logits_store.append(logits)
            targets = torch.sum(labels,dim=1)
            logits = torch.cat(logits_store)
            loss = lossfun(logits, labels)
            # _, predicted = torch.max(logits, 1)
            pred_sig = torch.sigmoid(logits)
            predicted = pred_sig.sum(dim=1).round()
            #Metric Calculation
            # total_dice+=MetricFun(outputs, labels)
            metrics_calc = metricfun(predicted.type(torch.int16),targets.type(torch.int16))
            running_loss += loss.data
            image_ids.extend(data["id"])
            # true_labels.extend(data["label"].numpy())
            true_labels.extend(targets.cpu().numpy())
            # pred_probs.extend(torch.nn.functional.softmax(logits.cpu(),dim=1).numpy())
            pred_probs.append(pred_sig.cpu().numpy())
            predicted_labels.extend(predicted.cpu().numpy())
    pred_probs = np.concatenate(pred_probs)
    metrics_calc = metricfun.compute()
    # print(metrics_calc)
    running_loss = running_loss.cpu().numpy()
    metricfun.reset()
    f1score = metrics_calc["test_F1Score"].cpu().numpy().item()
    test_accuracy = metrics_calc["test_Accuracy"].cpu().numpy()
    qwk = quadratic_kappa_coefficient(torch.tensor(predicted_labels).type(torch.int64),torch.tensor(true_labels).type(torch.int64))
    if verbosity:
        print("val accuracy: {}, val f1 score: {}, val qwk score: {}".format(test_accuracy,f1score,qwk))
    return qwk,running_loss/len(testloader),pred_probs,true_labels,image_ids

def train_full(datasets, model, args, verbosity=False, save_model=False, model_name=None):
    train_dataset, val_dataset = datasets
    samples_weights = train_dataset.get_sample_weights()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weights, len(samples_weights), replacement=True)
    trainloader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False,collate_fn=collate,sampler=sampler)
    valloader = torch.utils.data.DataLoader(val_dataset,batch_size=args.batch_size,shuffle=True,collate_fn=collate)
    DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(DEVICE)
    lossfunc = nn.BCEWithLogitsLoss().to(DEVICE)
    #optimizer
    optimizer = optim.Adam(model.parameters(),lr=args.lr/WARMUP_FACTOR, weight_decay = args.lamda)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs-WARMUP_EGO)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=WARMUP_FACTOR, total_epoch=WARMUP_EGO, after_scheduler=scheduler_cosine)
    #Metrics
    metricfunc = torchmetrics.MetricCollection([torchmetrics.Accuracy(num_classes=args.num_classes,average=None),
                                            torchmetrics.F1Score(num_classes=args.num_classes,average='weighted'),
                                            ])

    trainmetricdict = metricfunc.clone(prefix='train_').to(DEVICE)
    testmetricdict = metricfunc.clone(prefix='test_').to(DEVICE)
    best_metric = 0
    for epoch in range(1,args.num_epochs+1):
        train_one_epoch(model, trainloader ,optimizer, lossfunc, trainmetricdict, DEVICE, verbosity)
        test_metric,test_loss,pred_probs,true_labels,image_ids = val_one_epoch(model,valloader,lossfunc,testmetricdict,DEVICE,verbosity)
        # scheduler.step(test_loss)
        scheduler.step(epoch-1)
        if save_model:
            if test_metric>=best_metric:
                best_metric = test_metric
                torch.save(model.state_dict(),Path(args.save_dir)/f"{model_name}.pt")
    # print("Training done...")
    return test_metric,pred_probs,true_labels,image_ids 

def collate(batch):
    image = [ b['image'] for b in batch ] # w, h
    label = [ b['label'] for b in batch ]
    id = [ b['id'] for b in batch ]
    return {'image': image, 'label': torch.cat(label), 'id': np.array(id)}

def preprocess_data(args):
    ######## data preprocessing
    df = pd.read_csv(Path(args.csv_path)/"train.csv")
    # Find the duplicated cases
    duplicate_df = pd.read_csv(Path(args.csv_path)/"duplicate_imgids_imghash_thres_090.csv")
    duplicated = duplicate_df[duplicate_df['index_in_group'] != 0]
    duplicated_label = [x in duplicated.image_id.astype('string').to_list() for x in df.image_id.astype('string').to_list()]
    df['duplicated'] = duplicated_label
    clean_df = df.loc[df['duplicated'] == False, :]
    clean_df = clean_df.reset_index(drop=True)
    #Clean based on slides containing atleast >10 patches
    clean_idx = []
    data_store = []
    for i in tqdm(range(len(clean_df))):
        paths = clean_df.iloc[i]["image_id"]
        data = torch.load(Path(args.data_root_dir)/f"{paths}_featvec.pt",map_location="cpu")
        if len(data)<10:
            continue
        else:
            data_store.append(data)
            clean_idx.append(i)
    clean_df = clean_df.iloc[clean_idx,:]
    clean_df["data"] = data_store.copy()
    del data_store
    clean_df = clean_df.sort_values(by=["image_id"])
    clean_df["int_id"] = clean_df.index
    id_reference = clean_df[["int_id","image_id"]].copy()

    image_id_k_clean = clean_df.loc[clean_df.data_provider=='karolinska',:]
    image_id_r_clean = clean_df.loc[clean_df.data_provider=='radboud',:]
    X_k_clean, y_k_clean = image_id_k_clean[["int_id","image_id","data","data_provider","isup_grade"]], image_id_k_clean.isup_grade.astype('int32').to_list()
    X_r_clean, y_r_clean = image_id_r_clean[["int_id","image_id","data","data_provider","isup_grade"]], image_id_r_clean.isup_grade.astype('int32').to_list()

    #split full random
    # stratification based on ISUP grade
    X_train_val_k_clean, X_test_k_clean, y_train_val_k_clean, y_test_k_clean = train_test_split(X_k_clean, y_k_clean,
                                                        stratify=y_k_clean,
                                                        test_size=0.15, random_state=args.seed)

    X_train_k_clean, X_val_k_clean, y_train_k_clean, y_val_k_clean = train_test_split(X_train_val_k_clean, y_train_val_k_clean,
                                                        stratify=y_train_val_k_clean,
                                                        test_size=0.177, random_state=args.seed)

    X_train_val_r_clean, X_test_r_clean, y_train_val_r_clean, y_test_r_clean = train_test_split(X_r_clean, y_r_clean,
                                                        stratify=y_r_clean,
                                                        test_size=0.15, random_state=args.seed)

    X_train_r_clean, X_val_r_clean, y_train_r_clean, y_val_r_clean = train_test_split(X_train_val_r_clean, y_train_val_r_clean,
                                                        stratify=y_train_val_r_clean,
                                                        test_size=0.177, random_state=args.seed)

    X_train_clean = pd.concat((X_train_k_clean,X_train_r_clean))
    X_val_clean = pd.concat((X_val_k_clean,X_val_r_clean))
    X_test_clean = pd.concat((X_test_k_clean,X_test_r_clean))

    y_train_clean = y_train_k_clean + y_train_r_clean
    y_val_clean = y_val_k_clean + y_val_r_clean
    y_test_clean = y_test_k_clean + y_test_r_clean

    return (X_train_clean,y_train_clean),(X_val_clean,y_val_clean),(X_test_clean,y_test_clean), id_reference

def quadratic_kappa_coefficient(output, target):
    n_classes = 6
    output = torch.tensor(output)
    target = torch.tensor(target)

    try:
        target = F.one_hot(target, 6)
        output = F.one_hot(output, 6)
    except:
        print('last batch no data')
        return 0

    weights = torch.arange(0, n_classes, dtype=torch.float32, device=output.device) / (n_classes - 1)
    weights = (weights - torch.unsqueeze(weights, -1)) ** 2

    C = (output.t() @ target).t()  # confusion matrix

    hist_true = torch.sum(target, dim=0).unsqueeze(-1)
    hist_pred = torch.sum(output, dim=0).unsqueeze(-1)

    E = hist_true @ hist_pred.t()  # Outer product of histograms
    E = E / C.sum()  # Normalize to the sum of C.

    num = weights * C
    den = weights * E

    QWK = 1 - torch.sum(num) / torch.sum(den)
    return QWK


if __name__=="__main__":
    import argparse
    from mil_models import TransMIL_peg
    from panda_dataloader import Pandas_Dataset

    #hyperparameters settings
    parser = argparse.ArgumentParser(description='Configurations for Gleason Grading in PANDA dataset')
    parser.add_argument('--seed',type=int,default=1)
    parser.add_argument('--data_root_dir', type=str, default='../data/PANDAS/PANDAS_MIL_Patches_Selfpipeline_1MPP/', help='data directory')
    parser.add_argument('--csv_path', type=str, default='../data/PANDAS')
    parser.add_argument('--save_dir',type=str, default='../results/PANDAS')

    parser.add_argument('--num_classes',type=int, default=6)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--lamda', type=float, default=0.0005, help="weight decay to use in adam optimizer")
    parser.add_argument('--patience', type=int, default=10, help="number of epochs to wait in reducelronplateu lr scheduler")
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()

    #model defination
    model = TransMIL_peg(n_classes=args.num_classes-1,dim=512)
    (X_train_clean,y_train_clean),(X_val_clean,y_val_clean),(X_test_clean,y_test_clean), id_reference = preprocess_data(args)

    X_train_clean["isup_grade"] = y_train_clean
    train_split = X_train_clean.reset_index(drop=True)
    X_val_clean["isup_grade"] = y_val_clean
    val_split = X_val_clean.reset_index(drop=True)
    train_split = pd.concat((X_train_clean,X_val_clean)).reset_index(drop=True)
    X_test_clean["isup_grade"] = y_test_clean
    test_split = X_test_clean.reset_index(drop=True)

    trainset = Pandas_Dataset(train_split,args.data_root_dir)
    testset = Pandas_Dataset(test_split,args.data_root_dir)
    train_full((trainset,testset),model,args,verbosity=True,save_model=True,model_name=f"kagglepipeline_512_drpout_{time.strftime('_%d%b_%H_%M_%S', time.localtime())}")

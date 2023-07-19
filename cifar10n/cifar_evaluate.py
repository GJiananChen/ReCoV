import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import warnings
warnings.filterwarnings("ignore")
import random
import h5py

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedKFold, KFold
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

file_loc = Path(__file__).resolve().parent.parent
random.seed(1)
np.random.seed(1)
cifar10n_pt = str(file_loc / 'data/CIFAR-N/CIFAR-10_human.pt')
cifar_h5 = str(file_loc / 'data/CIFAR-N/cifar_feats.h5')

NOISE_TYPE = "aggre"
noise_file = torch.load(cifar10n_pt)
clean_label = noise_file['clean_label']
worst_label = noise_file['worse_label']
aggre_label = noise_file['aggre_label']
random_label1 = noise_file['random_label1']
random_label2 = noise_file['random_label2']
random_label3 = noise_file['random_label3']

# with open(str(file_loc / f"results/memory_cifarn_aggre_0.3.npy"),"rb") as file:
# with open(str(file_loc / f"results/memory_cifarn_worst_0.1_v2.npy"),"rb") as file:
with open(str(file_loc / f"results/memory_cifarn_aggre_0.1_v2.npy"),"rb") as file:
    memory = np.load(file)

identified = np.where(memory<=0.5)[0]
print(f"Number of identified samples: {len(identified)}")
with h5py.File(cifar_h5, "r") as f:
    X, X_test, y, y_test = f['train_feats'], f['test_feats'], f['train_labels'], f['test_labels']
    X, X_test, y, y_test = np.array(X), np.array(X_test), np.array(y), np.array(y_test)

if NOISE_TYPE == 'clean':
    y = clean_label
elif NOISE_TYPE == 'aggre':
    y = aggre_label
elif NOISE_TYPE == 'random1':
    y = random_label1
elif NOISE_TYPE == 'random2':
    y = random_label2
elif NOISE_TYPE == 'random3':
    y = random_label3
elif NOISE_TYPE == 'worst':
    y = worst_label
else:
    print('Noise type not recognized.')

X = pd.DataFrame(X)
X_test = pd.DataFrame(X_test)
y = pd.Series(y)
y_test = pd.Series(y_test)


subset_length = 50000
noisy = [0 if clean_label[x] == y[x] else 1 for x in range(subset_length)]
a, b = np.unique(y[:subset_length], return_counts=True)
print(b)
X = X[:subset_length]
y = y[:subset_length]

target_indice = identified

X1 = np.delete(X.to_numpy(), target_indice, axis=0)
y1 = np.delete(y.to_numpy(), target_indice, axis=0)
X1= pd.DataFrame(X1)
y1 = pd.Series(y1)

# np.array(clean_label[target_indice])
# np.array(aggre_label[target_indice])

random_state = 1
random.seed(random_state)
np.random.seed(random_state)
n_runs = 16000

# rf = RandomForestClassifier(n_jobs=-1)
rf = LogisticRegression()
print(f'--------Training RF model on noisy data')
rf.fit(X, y)

y_pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'ACC={acc*100:.3f}%')

# rf = RandomForestClassifier(n_jobs=-1)
rf = LogisticRegression()
print(f'--------Training RF model on cleaned data')
rf.fit(X1, y1)

y_pred = rf.predict(X_test)
pred_proba = rf.predict_proba(X_test)
acc = accuracy_score(y_test, y_pred)
print(f'ACC={acc*100:.3f}%')
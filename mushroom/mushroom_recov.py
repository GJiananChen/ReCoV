import os
import math
import pickle
import random
import os.path

import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import binom
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import KFold
from statsmodels.stats.multitest import multipletests
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, auc


def closest_value(input_list, input_value):
    difference = lambda input_list: abs(input_list - input_value)

    res = min(input_list, key=difference)

    return res

def pbinom(q,size,prob=0.5):
    """
    Calculates the cumulative of the binomial distribution
    """
    result=binom.cdf(k=q,n=size,p=prob,loc=0)
    return result

def qbinom(q, size, prob=0.5):
    """
    Calculates the quantile of the binomial distribution
    """
    result=binom.ppf(q=q,n=size,p=prob,loc=0)
    return result

def smallest_index(lst):
    return lst.index(min(lst))

def threshold_indice(lst, threshold=0.5):
    return [x[0] for x in enumerate(lst) if x[1] < threshold]

if __name__ == '__main__':
    n_runs = 5**7
    NOISE_RATIOS = [ 10, 20]
    for NOISE_RATIO in NOISE_RATIOS:
        random.seed(1)
        np.random.seed(1)

        mushroom = pd.read_csv('../data/mushrooms.csv')

        cols = mushroom.columns.to_list()
        cols.remove('class')

        mushroom = pd.get_dummies(mushroom, columns=cols)

        X = mushroom.drop(['class'], axis=1).copy()
        y = mushroom['class'].copy()
        y[y == 'p'] = 1
        y[y == 'e'] = 0
        y = y.astype('int').to_list()

        noisy = random.sample(list(enumerate(y)), int(0.01*NOISE_RATIO*len(y)))
        for i in noisy:
            index, label = i[0], i[1]
            y[index] = 1 - label
            y = pd.Series(y)

        all_candidates = np.array([], dtype='int')
        for random_state in range(1, n_runs+1):
            print(f'Current Seed: {random_state}')
            #Generate new splits for each run
            kf = KFold(n_splits=5, random_state=random_state,shuffle=True)
            aucs = []
            test_ids = []
            for fold, (train, test) in enumerate(kf.split(X,y)):
                X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train].to_list(), y.iloc[test].to_list()
                #train model
                lr = LogisticRegression()
                lr.fit(X_train, y_train)
                #Evaluate model
                y_pred = lr.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred)
                print(f'Fold: {fold}, ACC={acc:.3f}, AUC={auc:.3f}')
                aucs.append(auc)
                test_ids.append(test)

            #Find the worst fold
            candidates = test_ids[smallest_index(aucs)]
            all_candidates = np.append(all_candidates, candidates)

            #Display results at various time points
            if random_state == 5**3 or random_state == 5**4 or random_state == 5**5 or random_state == 5**6 or random_state == 5**7:
                print(all_candidates.shape)
                ids, counts = np.unique(all_candidates, return_counts=True)
                plt.clf()
                outlier_df = pd.DataFrame({'ids': ids, 'counts': counts})
                ax = sns.histplot(data=counts, stat='count')
                ax.xaxis.set_major_locator(ticker.MultipleLocator(400))
                plt.savefig('./results/mushroom_hist_NR5_78125.png')
                #Calculating threshold based on statistics
                upper, lower = random_state*1/5 + 2*math.sqrt(random_state*(5-1)/5**2), random_state*1/5 - 2*math.sqrt(random_state*(5-1)/5**2)
                print(np.where(counts > upper))

                identified = np.where(counts > upper)[0].tolist()
                identified2 = np.where(counts<lower)[0].tolist()

                gt = [x[0] for x in noisy]

                #Evaluate the confusion matrix for real noisy and identified noisy samples
                F = set(identified) #Predicted noisy
                G = set(range(0,len(y))) - set(gt)  #True clean
                F_t = set(range(0,len(y))) - set(identified) #Predicted clean
                M = set(gt) #True noisy
                ER1 = len(F.intersection(G))/len(G) #False positives
                ER2 = len(F_t.intersection(M))/len(M) #False negatives
                NEP = len(F.intersection(M))/ len(F) #True noisy identified

                with open(f'./results/mushroom_{random_state}_{NOISE_RATIO}_with_candidates.pkl','wb') as f:
                    pickle.dump(F, f)
                    pickle.dump(G, f)
                    pickle.dump(F_t, f)
                    pickle.dump(M, f)
                    pickle.dump(ER1, f)
                    pickle.dump(ER2, f)
                    pickle.dump(NEP, f)
                    pickle.dump(all_candidates, f)

        c_length = len(y)/5

        #Removing and plotting samples based on finding two seperate distributions
        noisy_labels = [1 if x in gt else 0 for x in list(range(8124))]
        for i in [100, 500, 1000, 5000, 10000, 15000, 20000, 50000, 78125]:
            ids, counts = np.unique(all_candidates[:int(c_length*i)], return_counts=True)
            pvalues = [1-pbinom(x, i, 0.2) for x in counts]
            reject, corrected_pvalues,_,_ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

            p = pvalues[np.where(corrected_pvalues == closest_value(corrected_pvalues, 0.05))[0][0]]
            corrected_threshold = qbinom(1-p, i, 0.2)

            outlier_df = pd.DataFrame({'ids': ids, 'counts': counts, 'pvalues': pvalues,
                                       'corrected': corrected_pvalues, 'noisy_label':noisy_labels})
            if not os.path.exists(f'./figures/mushroom/NR{NOISE_RATIO}'):
                os.mkdir(f'./figures/mushroom/NR{NOISE_RATIO}')

            plt.clf()
            ax = sns.histplot(data=outlier_df, x='pvalues', stat='count', multiple="stack", kde=False,
             palette="pastel", hue="noisy_label",
             element="bars", legend=True)
            plt.legend(title='Noisy', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0)
            plt.xlabel('Number of recurrance')
            plt.ylabel('Number of samples')
            plt.savefig(f'./figures/mushroom/NR{NOISE_RATIO}/mushroom_hist_NR{NOISE_RATIO}_{i}_pvalues.png')


            plt.clf()
            ax = sns.histplot(data=outlier_df, x='corrected', stat='count', multiple="stack", kde=False,
             palette="pastel", hue="noisy_label",
             element="bars", legend=True)
            plt.legend(title='Noisy', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0)
            plt.xlabel('Number of recurrance')
            plt.ylabel('Number of samples')
            plt.savefig(f'./figures/mushroom/NR{NOISE_RATIO}/mushroom_hist_NR{NOISE_RATIO}_{i}_mtc_pvalues.png')

            plt.clf()
            ax = sns.histplot(data=outlier_df, x='counts', stat='count', multiple="stack", kde=False,
             palette="pastel", hue="noisy_label",
             element="bars", legend=True)
            plt.legend(title='Noisy', loc='best', labels=['Noisy', 'Clean'])
            ax.xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(i/100)))
            upper, lower = i*1/5 + 2*math.sqrt(i*(5-1)/5**2), i*1/5 - 2*math.sqrt(i*(5-1)/5**2)
            plt.axvline(int(corrected_threshold), 0, 1)
            plt.axvline(int(upper), 0, 1, linestyle='--')
            plt.xlabel('Number of recurrance')
            plt.ylabel('Number of samples')
            # ax.axvline(int(upper))
            plt.savefig(f'./figures/mushroom/NR{NOISE_RATIO}/mushroom_hist_NR5_{i}_counts.svg', format='svg', dpi=600)
            plt.savefig(f'./figures/mushroom/NR{NOISE_RATIO}/mushroom_hist_NR5_{i}_counts.png', dpi=600)

        identified = np.where(counts > corrected_threshold)[0].tolist()

        F = set(identified) #Predicted noisy
        G = set(range(0,len(y))) - set(gt)  #True clean
        F_t = set(range(0,len(y))) - set(identified) #Predicted clean
        M = set(gt) #True noisy
        ER1 = len(F.intersection(G))/len(G) #False positives
        ER2 = len(F_t.intersection(M))/len(M) #False negatives
        NEP = len(F.intersection(M))/ len(F) #True noisy identified
        
        #Removing samples identified as noisy and evaluating the predictions of the model
        y = y.loc[list(F_t)]
        X = X.loc[F_t]

        aucs = []
        accs = []
        kf = KFold(n_splits=5, random_state=random_state, shuffle=True)
        test_ids = []
        for fold, (train, test) in enumerate(kf.split(X, y)):
            X_train, X_test, y_train, y_test = X.iloc[train], X.iloc[test], y.iloc[train].to_list(), y.iloc[
                test].to_list()
            lr = LogisticRegression()
            lr.fit(X_train, y_train)

            y_pred = lr.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_pred)
            print(f'Fold: {fold}, ACC={acc:.3f}, AUC={auc:.3f}')
            aucs.append(auc)
            accs.append(acc)
            test_ids.append(test)
        print(sum(aucs) / len(aucs))
        print(sum(accs) / len(accs))

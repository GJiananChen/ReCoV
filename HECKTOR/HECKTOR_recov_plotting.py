import random
import math
import pickle
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker
from statsmodels.stats.multitest import multipletests
from scipy.stats import binom
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
# import libraries (some are for cosmetics)
from sklearn.mixture import GaussianMixture as GMM
import matplotlib as mpl

mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams.update({'font.size': 15, 'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix'})

def string_to_list(text):
    text = text.replace('\n', '')
    text = text.strip('][')
    text = text.replace(']', '')
    text = text.replace('[', '')
    text = text.split(' ')
    lst = [int(x) for x in text if x!='']
    return lst

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

def pnorm(q, mean, std):
    result = norm.cdf(q, loc=mean, scale=std)
    return result

def qnorm(q, mean, std):
    result = norm.ppf(q, loc=mean, scale=std)
    return result


def smallest_index(lst):
    return lst.index(min(lst))

def calculate_txt_indices(txtfile):
    with open(txtfile, 'r') as out:
        a = out.readlines()
        set_seed_line_ind = []

        for ind, line in enumerate(a):
                if 'Random seed set as' in line:
                    set_seed_line_ind.append(ind)

    return set_seed_line_ind

def split_txt_files(set_seed_line_ind, output_folder, txtfile):
    small_filename = f'small_file_310.txt'
    smallfile = open(os.path.join(output_folder, small_filename), "w")
    with open(txtfile) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno in set_seed_line_ind:
                if smallfile:
                    smallfile.close()
                small_filename = f'small_file_{lineno}.txt'
                smallfile = open(os.path.join(output_folder, small_filename), "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

def smallest_index(lst):
    return lst.index(min(lst))

def threshold_indice(lst, threshold=0.5):
    return [x[0] for x in enumerate(lst) if x[1] < threshold]

if __name__ == '__main__':
    NR = 0
    output_path = r'E:\PycharmProjects\AMINN_torch_dev\figures\hecktor'
    figure_path = os.path.join(output_path, f'AMINN')
    data = []
    with open(f'./pickle/hecktor_7_1000.pkl', 'rb') as f:
        while True:
            try:
                data.append(pickle.load(f))
            except EOFError:
                break

    data2 = []
    with open(f'./pickle/hecktor_8_1000.pkl', 'rb') as f:
        while True:
            try:
                data2.append(pickle.load(f))
            except EOFError:
                break

    data3 = []
    with open(f'./pickle/hecktor_9_1000.pkl', 'rb') as f:
        while True:
            try:
                data3.append(pickle.load(f))
            except EOFError:
                break

    data4 = []
    with open(f'./pickle/hecktor_10_1000.pkl', 'rb') as f:
        while True:
            try:
                data4.append(pickle.load(f))
            except EOFError:
                break

    all_candidates = data[0]
    last_aucs = data[1]

    all_candidates += data2[0]
    last_aucs += data2[1]

    all_candidates += data3[0]
    last_aucs += data3[1]

    all_candidates += data4[0]
    last_aucs += data4[1]

    n_runs = len(all_candidates)

    random.seed(1)
    np.random.seed(1)
    all_candidates = list(itertools.chain.from_iterable(all_candidates))
    c_length = float(len(all_candidates)/n_runs)
    for i in [4000]:
        ids, counts = np.unique(all_candidates[:int(c_length * i)], return_counts=True)
        pvalues = [1 - pbinom(x, i, 0.2) for x in counts]
        reject, corrected_pvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

        p = pvalues[np.where(corrected_pvalues == closest_value(corrected_pvalues, 0.05))[0][0]]
        corrected_threshold = qbinom(1 - p, i, 0.2)

        outlier_df = pd.DataFrame({'ids': ids, 'counts': counts, 'pvalues': pvalues,
                                   'corrected': corrected_pvalues})

        plt.clf()
        ax = sns.histplot(data=outlier_df, x='pvalues', stat='count', multiple="stack", kde=False,
                          palette="pastel",
                          element="bars", legend=True)
        if NR == 0:
            plt.legend(title='Noisy GT', loc='best', labels=['Clean'], borderaxespad=0)
        else:
            plt.legend(title='Noisy GT', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0)
        plt.savefig(os.path.join(figure_path, f'hn_hist_NR{NR}_{i}_pvalues.png'))

        plt.clf()
        ax = sns.histplot(data=outlier_df, x='corrected', stat='count', multiple="stack", kde=False,
                          palette="pastel",
                          element="bars", legend=True)
        if NR == 0:
            plt.legend(title='Noisy GT', loc='best', labels=['Clean'], borderaxespad=0)
        else:
            plt.legend(title='Noisy GT', loc='best', labels=['Noisy', 'Clean'], borderaxespad=0)
        plt.savefig(os.path.join(figure_path, f'hn_hist_NR{NR}_{i}_mtc_pvalues.png'))

        plt.clf()
        ax = sns.histplot(data=outlier_df, x='counts', stat='count', multiple="stack", kde=False,
                          palette="pastel", binwidth=25, binrange=(500, 1350),
                          element="bars", legend=True)
        if NR == 0:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(i / 40)))
        else:
            ax.xaxis.set_major_locator(ticker.MultipleLocator(math.ceil(i / 100)))
        upper, lower = i * 1 / 5 + 2 * math.sqrt(i * (5 - 1) / 5 ** 2), i * 1 / 5 - 2 * math.sqrt(
            i * (5 - 1) / 5 ** 2)
        plt.axvline(int(corrected_threshold), 0, 1, color='r')
        plt.title(f'HECKTOR: n_runs={i}', fontsize=22)
        plt.xlabel('Number of recurrances', fontsize=18)
        plt.ylabel('Number of samples', fontsize=18)
        plt.xlim((400, 1450))
        plt.xticks(np.arange(400, 1450, 100))
        # ax.axvline(int(upper))
        plt.savefig(os.path.join(figure_path, f'hn_hist_NR{NR}_{i}_counts.svg'), format='svg', dpi=600)
        plt.savefig(os.path.join(figure_path, f'hn_hist_NR{NR}_{i}_counts.png'), dpi=600)
        plt.savefig(os.path.join(figure_path, f'hecktor_hist_NR{NR}_{i}_counts.svg'), format='svg', dpi=600)
        plt.savefig(os.path.join(figure_path, f'hecktor_hist_NR{NR}_{i}_counts.png'), dpi=600)

    plt.clf()
    x = np.reshape(counts,(-1,1))
    bics = []
    min_bic = 0
    counter = 1
    for i in range(10):  # test the AIC/BIC metric between 1 and 10 components
        gmm = GMM(n_components=counter, max_iter=1000, random_state=0, covariance_type='full')
        labels = gmm.fit(x).predict(x)
        bic = gmm.bic(x)
        bics.append(bic)
        if bic < min_bic or min_bic == 0:
            min_bic = bic
            opt_bic = counter
        counter = counter + 1

    #hn (101,113)
    # plot the evolution of BIC/AIC with the number of components
    fig = plt.figure(figsize=(16, 6))
    ax = fig.add_subplot(1, 2, 1)
    # Plot 1
    plt.plot(np.arange(1, 11), bics, 'o-', lw=3, c='black', label='BIC')
    plt.legend(frameon=False, fontsize=20)
    plt.xlabel('Number of components', fontsize=22)
    plt.ylabel('Information criterion', fontsize=22)
    plt.xticks(np.arange(0, 11, 2))
    plt.title('Opt. components = ' + str(opt_bic), fontsize=26)


    # Since the optimal value is n=2 according to both BIC and AIC, let's write down:
    n_optimal = opt_bic

    # n_optimal = 2
    # create GMM model object
    gmm = GMM(n_components=n_optimal, max_iter=1000, random_state=10, covariance_type='full')

    # find useful parameters
    mean = gmm.fit(x).means_
    covs = gmm.fit(x).covariances_
    weights = gmm.fit(x).weights_

    # Since the optimal value is n=2 according to both BIC and AIC, let's write down:
    n_optimal = opt_bic

    if n_optimal == 1:
        x_axis = np.arange(400, 1425, 25)
        y_axis0 = norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0]))) * weights[0]
        ax = fig.add_subplot(1, 2, 2)
        # Plot 2
        plt.hist(x, density=True, color='black', bins=np.arange(400, 1400, 25))
        plt.plot(x_axis, y_axis0, lw=3, c='C0')
        plt.xlim(400, 1400)
        plt.xticks(np.arange(400, 1450, 100))
        # plt.ylim(0.0, 0.005)
        plt.xlabel(r"X", fontsize=22)
        plt.ylabel(r"Density", fontsize=22)

        plt.subplots_adjust(wspace=0.4)
        plt.savefig('BIC_hecktor.png', DPI=300)
        plt.show()
        plt.close('all')
    elif n_optimal == 2:
        x_axis = np.arange(400, 1425, 25)
        y_axis0 = norm.pdf(x_axis, float(mean[0][0]), np.sqrt(float(covs[0][0][0]))) * weights[0]  # 1st gaussian
        y_axis1 = norm.pdf(x_axis, float(mean[1][0]), np.sqrt(float(covs[1][0][0]))) * weights[1]  # 2nd gaussian

        ax = fig.add_subplot(1, 2, 2)
        # Plot 2
        plt.hist(x, density=True, color='black', bins=np.arange(400, 1400, 25))
        plt.plot(x_axis, y_axis0, lw=3, c='C0')
        plt.plot(x_axis, y_axis1, lw=3, c='C1')
        plt.plot(x_axis, y_axis0 + y_axis1, lw=3, c='C2', ls='dashed')
        plt.xlim(400, 1400)
        plt.xticks(np.arange(400, 1450, 100))
        # plt.ylim(0.0, 0.005)
        plt.xlabel(r"X", fontsize=22)
        plt.ylabel(r"Density", fontsize=22)

        plt.subplots_adjust(wspace=0.4)
        plt.savefig('BIC_hecktor.png', DPI=300)
        plt.close('all')
    else:
        print(n_optimal)

    std = math.sqrt(covs[0][0][0])
    mu = mean[0][0]
    upper = mu+2*std

    pvalues = [1 - pnorm(x, mu, std) for x in counts]
    reject, corrected_pvalues, _, _ = multipletests(pvalues, alpha=0.05, method='fdr_bh')

    p = pvalues[np.where(corrected_pvalues == closest_value(corrected_pvalues, 0.05))[0][0]]
    corrected_threshold = qnorm(1 - p, mean, std)

    outlier_df = pd.DataFrame({'ids': ids, 'counts': counts, 'pvalues': pvalues,
                               'corrected': corrected_pvalues})


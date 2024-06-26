"""Pytorch dataset object that loads MNIST dataset as bags."""
import sys
from pathlib import Path
file_loc = Path(__file__).resolve().parent.parent
sys.path.append(str(file_loc))

import numpy as np
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from utils import create_lung_df, create_hn_df, create_liver_df, feature_sets_lung, feature_sets_hn, feature_sets_liver
import pandas as pd


class AMINNDataset:
    def __init__(self, data='liver', censor=0, subset='multi',exclusion=[]):
        self.data = data
        self.exclusion = exclusion
        # self.censor = censor
        # self.subset = subset

    def get_csv(self):
        if self.data == 'liver':
            self.data_csv = str(file_loc / 'data/liver_1.5mm.csv')
            self.cli_csv = str(file_loc / 'data/GadClinicalInfo.csv')
        elif self.data == 'hn':
            self.data_csv = str(file_loc / 'data/HN_multi.csv')
            self.cli_csv = str(file_loc / 'data/HN_clinical.csv')
        elif self.data == 'hn_simulation':
            self.data_csv = str(file_loc / 'data/HN_multi.csv')
            self.cli_csv = str(file_loc / 'data/HN_clinical_simulation.csv')
        elif self.data == 'lung':
            self.data_csv = str(file_loc / 'data/Lung_multi.csv')
            self.cli_csv = str(file_loc / 'data/Lung_clinical.csv')
        elif self.data == 'hecktor':
            self.data_csv = str(file_loc / 'data/HECKTOR_train_all_setting2.csv')
            self.cli_csv = str(file_loc / 'data/hecktor2022_endpoint_training.csv')
        elif self.data == 'hecktor_train':
            self.data_csv = str(file_loc / 'data/HECKTOR_train_train_setting2.csv')
            self.cli_csv = str(file_loc / 'data/hecktor2022_endpoint_training.csv')
        elif self.data == 'hecktor_test':
            self.data_csv = str(file_loc / 'data/HECKTOR_train_test_setting2.csv')
            self.cli_csv = str(file_loc / 'data/hecktor2022_endpoint_training.csv')
        else:
            print('dataset not supported')
            self.data_csv = None
            self.cli_csv = None



    def preprocess_df(self, censor=0):
        self.df = pd.read_csv(self.data_csv)
        cli = pd.read_csv(self.cli_csv)
        if self.data == 'liver':
            self.df = self.df.rename(columns={'pid':'ID'})
            cli.columns = ['ID', 'age', 'sex', 'fong', 'mor', 'fu']
            cli = cli.loc[:, ['ID', 'mor', 'fu']]
        elif self.data == 'hn':
            self.df.ID = [x.split('_')[0] for x in self.df.ID]
            cli = cli.loc[:, ['id', 'event_overall_survival', 'overall_survival_in_days', 'event_distant_metastases',
                              'distant_metastases_in_days']]
            cli.columns = ['ID', 'mor', 'fu', 'dm', 'dmfu']
        elif self.data == 'hn_simulation':
            self.df.ID = [x.split('_')[0] for x in self.df.ID]
            cli = cli.loc[:, ['id', 'event_overall_survival', 'overall_survival_in_days', 'event_distant_metastases',
                              'distant_metastases_in_days']]
            cli.columns = ['ID', 'mor', 'fu', 'dm', 'dmfu']
        elif self.data == 'lung':
            self.df.ID = [x.split('_')[0] for x in self.df.ID]
            cli = cli.loc[:, ['PatientID', 'deadstatus.event', 'Survival.time']]
            cli.columns = ['ID', 'mor', 'fu']
        elif self.data == 'hecktor' or self.data == 'hecktor_train' or self.data=='hecktor_test':
            self.df.ID = [x.split('_')[0] for x in self.df.ID]
            cli = cli.loc[:, ['PatientID', 'Relapse', 'RFS']]
            cli.columns = ['ID', 'mor', 'fu']
        else:
            print('Dataset not supported.')
        if censor != 0:
            for i in range(len(cli)):
                if cli['fu'][i] > censor:
                    cli.iloc[i, 1] = 0
                    cli.iloc[i, 2] = censor
        self.df = pd.merge(self.df, cli, on='ID', how='inner')

    def get_subsets(self, subset='all'):
        if self.data == 'hn' or self.data == 'lung' or self.data =='hn_simulation':
            self.subdf = self.df.sort_values('original_shape_MeshVolume', ascending=False)
        elif self.data == 'hecktor' or self.data =='hecktor_train' or self.data =='hecktor_test':
            self.subdf = self.df.sort_values('original_shape_MeshVolume_x', ascending=False)
        else:
            self.subdf = self.df.sort_values('original_shape_Volume', ascending=False)
        self.n_lesions = self.subdf['ID'].value_counts()
        multifocal = self.n_lesions.index[self.n_lesions > 1]
        multifocal_idx = [x in multifocal for x in self.subdf['ID']]
        if self.data == 'hn' or self.data == 'lung' or self.data == 'hn_simulation':
            primary_idx = ['GTV-1_1' in x for x in self.subdf['Mask']]

        if self.data== 'liver':
            self.subdf.ID = self.subdf.ID
        elif self.data== 'hn' or self.data=='hn_simulation':
            self.subdf.ID = [x[2:] for x in self.subdf.ID]
        elif self.data== 'lung':
            self.subdf.ID = [x[6:] for x in self.subdf.ID]
        elif self.data== 'hecktor' or self.data=='hecktor_train' or self.data=='hecktor_test':
            self.subdf.ID = self.subdf.ID
        else:
            print('Dataset not supported')

        if subset == 'largest':
            self.subdf = self.subdf.drop_duplicates(subset='ID', keep='first')
        elif subset == 'multi':
            self.subdf = self.subdf[multifocal_idx]
        elif subset == 'uni':
            self.subdf = self.subdf[~np.array(multifocal_idx)]
        elif subset == 'multi_largest':
            self.subdf = self.subdf[multifocal_idx].drop_duplicates(subset='ID', keep='first')
        elif subset == 'all':
            self.subdf = self.subdf
        elif subset == 'primary':
            self.subdf = self.subdf[primary_idx]
        elif subset == 'outlier_removed':
            temp = set(np.arange(len(self.subdf)))
            self.subdf = self.subdf.iloc[list(temp-set(self.exclusion))]
        else:
            print("Supported subsets are uni, multi, largest, all, primary and multi_largest")

    def get_features(self, feature_class='original', normalize=True):
        # Keep original features only and discard Laplacian of Gaussian (Log) features and wavelet features
        idxs = self.subdf.columns.str.contains(r'sigma|wavelet|diagnostics')
        if feature_class == 'original':
            self.subdf = self.subdf.iloc[:, ~idxs]
        if self.data=='hecktor' or self.data=='hecktor_train' or self.data=='hecktor_test':
            first_idx = np.where(self.subdf.columns.str.contains('original_shape_Elongation_x'))[0].item()
            last_idx = np.where(self.subdf.columns.str.contains('mor'))[-1].item()
            self.subdf['Gender'].replace('M', 1., inplace=True)
            self.subdf['Gender'].replace('F', 0., inplace=True)
        else:
            first_idx = np.where(self.subdf.columns.str.contains('original_shape_Elongation'))[0].item()
            last_idx = np.where(self.subdf.columns.str.contains('mor'))[-1].item()
        np.where(self.subdf.columns)
        # Hard-coded feature columns, should be changed based on specific datasets
        X = self.subdf.iloc[:, first_idx: last_idx]
        # Two step normalization
        if normalize:
            # Log transformation with a correction term
            X = X - X.min() + (X - X.min()).median()
            X = np.log(X)
        self.X = (X - X.mean()) / X.std()
        # xmin = pd.read_csv('xmin.csv',index_col=0)
        # xmedian = pd.read_csv('xmedian.csv',index_col=0)
        # xmean = pd.read_csv('xmean.csv',index_col=0)
        # xstd = pd.read_csv('std.csv',index_col=0)
        # X = X-xmin.squeeze()+xmedian.squeeze()
        # X = np.log(X)
        # self.X = (X-xmean.squeeze())/xstd.squeeze()
        # Column 0: Patient ID, column -2: follow up time and column -1, mortality were passed as bag labels
        if self.data == 'liver':
            self.Y = self.subdf.iloc[:, [0, -2, -1]]
        elif self.data == 'hn' or self.data=='hn_simulation':
            self.Y = self.subdf.iloc[:, [1, -4, -3, -2, -1]]
        elif self.data== 'lung':
            self.Y = self.subdf.iloc[:, [1, -2, -1]]
        elif self.data == 'hecktor' or self.data == 'hecktor_train' or self.data == 'hecktor_test':
            self.Y = self.subdf.iloc[:, [2, -2, -1]]
        else:
            print('Dataset not supported')


    def extract_dataset(self, subset='all', censor=0, feature_class='original', normalize=True):
        self.get_csv()
        self.preprocess_df(censor=censor)
        self.get_subsets(subset=subset)
        self.get_features(feature_class=feature_class, normalize=normalize)
        return self.X, self.Y


def generate_dataset(data_csv, cli_csv, data='lung', subset = 'multi_largest', censor=1095, normalize=True):
    if data == 'liver':
        df = create_liver_df(data_csv, cli_csv,censor=censor, subset=subset)
        features, labels = feature_sets_liver(df, normalize=normalize)
    elif data == 'hn' or 'hn_simulation':
        # normalize=True if applying two-step normalization
        df = create_hn_df(data_csv, cli_csv,censor=censor, subset=subset)
        features, labels = feature_sets_hn(df, normalize=normalize)
    elif data == 'lung':
        # normalize=True if applying two-step normalization
        df = create_lung_df(data_csv, cli_csv,censor=censor, subset=subset)
        features, labels = feature_sets_lung(df, normalize=normalize)
    else:
        print('dataset not supported')
    return features, labels

def df_to_dataset(X, Y, nfolds, shuffle=True):
    bags_nm = np.asarray(Y['ID'], dtype=str)
    bags_label = np.asarray(Y['mor'], dtype='float64')
    ins_fea = np.asarray(X, dtype='float64')

    ins_idx_of_input = {}
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input:
            ins_idx_of_input[bag_nm].append(id)
        else:
            ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
            bag_fea[2].append(bag_nm)
        bags_fea.append(bag_fea)

    skf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=1234)
    datasets = []
    bags_list = [item[1] for item in bags_fea]
    bag_label = [item[0] for item in bags_list]
    for train_idx, test_idx in skf.split(bags_fea, bag_label):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)

    return datasets

def df_to_dataset_reg(X, Y, nfolds, shuffle=True):
    bags_nm = np.asarray(Y['ID'], dtype=str)
    bags_label = np.asarray(Y['fu'], dtype='float32')
    bag_event = np.asarray(Y['mor'], dtype='float32')
    ins_fea = np.asarray(X)

    ins_idx_of_input = {}
    for id, bag_nm in enumerate(bags_nm):
        if bag_nm in ins_idx_of_input:
            ins_idx_of_input[bag_nm].append(id)
        else:
            ins_idx_of_input[bag_nm] = [id]
    bags_fea = []
    for bag_nm, ins_idxs in ins_idx_of_input.items():
        bag_fea = ([], [], [], [])
        for ins_idx in ins_idxs:
            bag_fea[0].append(ins_fea[ins_idx])
            bag_fea[1].append(bags_label[ins_idx])
            bag_fea[2].append(bag_event[ins_idx])
            bag_fea[3].append(bag_nm)
        bags_fea.append(bag_fea)

    skf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=1234)
    datasets = []
    bags_list = [item[2] for item in bags_fea]
    bag_event = [item[0] for item in bags_list]
    for train_idx, test_idx in skf.split(bags_fea, bag_event):
        dataset = {}
        dataset['train'] = [bags_fea[ibag] for ibag in train_idx]
        dataset['test'] = [bags_fea[ibag] for ibag in test_idx]
        datasets.append(dataset)

    return datasets


class MultiFocalBags(data_utils.Dataset):
    def __init__(self, X, Y, seed=1):
        self.X = X
        self.Y = Y
        self.seed = seed

        self.bags_list, self.labels_list, self.bags_nm_list = self._create_bags()

    def _create_bags(self):
        bags_nm = np.asarray(self.Y['ID'], dtype=str)
        bags_label = np.asarray(self.Y['mor'], dtype='float32')
        ins_fea = np.asarray(self.X)

        ins_idx_of_input = {}
        for id, bag_nm in enumerate(bags_nm):
            if bag_nm in ins_idx_of_input:
                ins_idx_of_input[bag_nm].append(id)
            else:
                ins_idx_of_input[bag_nm] = [id]
        bags_fea = []
        for bag_nm, ins_idxs in ins_idx_of_input.items():
            bag_fea = ([], [], [])
            for ins_idx in ins_idxs:
                bag_fea[0].append(ins_fea[ins_idx])
                bag_fea[1].append(bags_label[ins_idx])
                bag_fea[2].append(bag_nm)
            bags_fea.append(bag_fea)

        bags_list = [item[0] for item in bags_fea]
        labels_list = [item[1] for item in bags_fea]
        bags_nm_list = [item[2] for item in bags_fea]
        return bags_list, labels_list, bags_nm_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = max(self.labels_list[index])
        bag_nm = self.bags_nm_list[index]
        return bag, label, bag_nm


class MultiFocalRegBags(data_utils.Dataset):
    def __init__(self, X, Y, seed=1):
        self.X = X
        self.Y = Y
        self.seed = seed
        self.bags_list, self.labels_list, self.bags_nm_list, self.bags_fu_list = self._create_bags()

    def _create_bags(self):
        bags_nm = np.asarray(self.Y['ID'], dtype=str)
        bags_label = np.asarray(self.Y['mor'], dtype='float32')
        bags_fu = np.asarray(self.Y['fu'], dtype='int32')
        ins_fea = np.asarray(self.X)

        ins_idx_of_input = {}
        for id, bag_nm in enumerate(bags_nm):
            if bag_nm in ins_idx_of_input:
                ins_idx_of_input[bag_nm].append(id)
            else:
                ins_idx_of_input[bag_nm] = [id]
        bags_fea = []
        for bag_nm, ins_idxs in ins_idx_of_input.items():
            bag_fea = ([], [], [], [])
            for ins_idx in ins_idxs:
                bag_fea[0].append(ins_fea[ins_idx])
                bag_fea[1].append(bags_label[ins_idx])
                bag_fea[2].append(bag_nm)
                bag_fea[3].append(bags_fu[ins_idx])
            bags_fea.append(bag_fea)

        bags_list = [item[0] for item in bags_fea]
        labels_list = [item[1] for item in bags_fea]
        bags_nm_list = [item[2] for item in bags_fea]
        bags_fu_list = [item[3] for item in bags_fea]
        return bags_list, labels_list, bags_nm_list, bags_fu_list

    def __len__(self):
        return len(self.labels_list)

    def __getitem__(self, index):
        bag = self.bags_list[index]
        label = max(self.labels_list[index])
        bag_nm = self.bags_nm_list[index]
        bag_fu = max(self.bags_fu_list[index])
        return bag, label, bag_nm, bag_fu, index

class MnistRegBags(data_utils.Dataset):
    def __init__(self, mean_bag_length=10, var_bag_length=2, num_bag=250, seed=1, train=True):
        self.mean_bag_length = mean_bag_length
        self.var_bag_length = var_bag_length
        self.num_bag = num_bag
        self.train = train

        self.r = np.random.RandomState(seed)

        self.num_in_train = 60000
        self.num_in_test = 10000

        if self.train:
            self.train_bags_list, self.train_labels_list = self._create_bags()
        else:
            self.test_bags_list, self.test_labels_list = self._create_bags()

    def _create_bags(self):
        if self.train:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=True,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_train,
                                           shuffle=False)
        else:
            loader = data_utils.DataLoader(datasets.MNIST('../datasets',
                                                          train=False,
                                                          download=True,
                                                          transform=transforms.Compose([
                                                              transforms.ToTensor(),
                                                              transforms.Normalize((0.1307,), (0.3081,))])),
                                           batch_size=self.num_in_test,
                                           shuffle=False)

        for (batch_data, batch_labels) in loader:
            all_imgs = batch_data
            all_labels = batch_labels

        bags_list = []
        labels_list = []

        for i in range(self.num_bag):
            bag_length = np.int(self.r.normal(self.mean_bag_length, self.var_bag_length, 1))
            if bag_length < 1:
                bag_length = 1

            if self.train:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_train, bag_length))
            else:
                indices = torch.LongTensor(self.r.randint(0, self.num_in_test, bag_length))

            labels_in_bag = all_labels[indices]
            # labels_in_bag = labels_in_bag == self.target_number

            bags_list.append(all_imgs[indices])
            labels_list.append(labels_in_bag)

        return bags_list, labels_list

    def __len__(self):
        if self.train:
            return len(self.train_labels_list)
        else:
            return len(self.test_labels_list)

    def __getitem__(self, index):
        if self.train:
            bag = self.train_bags_list[index]
            label = [sum(self.train_labels_list[index]), self.train_labels_list[index]]
        else:
            bag = self.test_bags_list[index]
            label = [sum(self.test_labels_list[index]), self.test_labels_list[index]]
        # label[1] = label[1] * label[1] * 0.1
        label[1] = label[1] + torch.rand(label[1].shape) - 0.5
        label[0] = sum(label[1])
        return bag, label



if __name__ == "__main__":

    train_loader = data_utils.DataLoader(MnistRegBags(mean_bag_length=5,
                                                   var_bag_length=2,
                                                   num_bag=500,
                                                   seed=1,
                                                   train=True),
                                         batch_size=1,
                                         shuffle=True)

    test_loader = data_utils.DataLoader(MnistRegBags(mean_bag_length=5,
                                                  var_bag_length=2,
                                                  num_bag=500,
                                                  seed=1,
                                                  train=False),
                                        batch_size=1,
                                        shuffle=False)

    len_bag_list_train = []
    mnist_bags_train = 0
    for batch_idx, (bag, label) in enumerate(train_loader):
        len_bag_list_train.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_train += label[0].numpy()[0]
    print('Number positive train bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_train, len(train_loader),
        np.mean(len_bag_list_train), np.max(len_bag_list_train), np.min(len_bag_list_train)))

    len_bag_list_test = []
    mnist_bags_test = 0
    for batch_idx, (bag, label) in enumerate(test_loader):
        len_bag_list_test.append(int(bag.squeeze(0).size()[0]))
        mnist_bags_test += label[0].numpy()[0]
    print('Number positive test bags: {}/{}\n'
          'Number of instances per bag, mean: {}, max: {}, min {}\n'.format(
        mnist_bags_test, len(test_loader),
        np.mean(len_bag_list_test), np.max(len_bag_list_test), np.min(len_bag_list_test)))
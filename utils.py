import numpy as np
import pandas as pd
import ast
# def convertToBatch(bags):
#     data_set = []
#     for ibag, bag in enumerate(bags):
#         batch_data = np.asarray(bag[0], dtype='float32')
#         batch_label = np.asarray(bag[1])
#         batch_id = np.asarray(bag[2], dtype='int16')
#         data_set.append((batch_data, batch_label, batch_id))
#     return data_set
#
# def convert_to_batch_reg(bags):
#     data_set = []
#     for ibag, bag in enumerate(bags):
#         batch_data = np.asarray(bag[0], dtype='float32')
#         batch_label = np.asarray(bag[1])
#         batch_event = np.asarray(bag[2])
#         batch_id = np.asarray(bag[3], dtype='int16')
#         data_set.append((batch_data, batch_label, batch_event, batch_id))
#     return data_set

def create_liver_df(data_csv, cli_csv, censor=0, subset='multi'):
    df = pd.read_csv(data_csv)
    cli = pd.read_csv(cli_csv)
    cli.columns = ['pid', 'age', 'sex', 'fong', 'mor', 'fu']
    if censor != 0:
        for i in range(len(cli)):
            if cli['fu'][i] > censor:
                cli.iloc[i, 4] = 0
                cli.iloc[i, 5] = censor
    df = pd.merge(df, cli, on='pid', how='inner')
    df = df.sort_values('original_shape_Volume', ascending=False)
    n_lesions = df['pid'].value_counts()
    multifocal = n_lesions.index[n_lesions>1]
    multifocal_idx = [x in multifocal for x in df['pid']]

    if subset == 'largest':
        df = df.drop_duplicates(subset='pid', keep='first')
        return df
    elif subset == 'multi':
        return df[multifocal_idx]
    elif subset == 'uni':
        return df[~np.array(multifocal_idx)]
    elif subset == 'multi_largest':
        return df[multifocal_idx].drop_duplicates(subset='pid', keep='first')
    elif subset == 'all':
        return df
    else:
        print("Supported subsets are uni, multi and largest")

def feature_sets_liver(df, feature_class='original', normalize = True):
    idxs = df.columns.str.contains(r'sigma|wavelet')
    if feature_class == 'original':
        df = df.iloc[:, ~idxs]
    X = df.iloc[:, 15: -6]
    X = X.drop(columns = 'original_firstorder_Minimum')
    if normalize:
        X = X - X.min() + X.median()
        X = np.log(X)
    X = (X - X.mean()) / X.std()
    Y = df.iloc[:,[0,-2,-1]]
    return X, Y

def create_hn_df(data_csv, cli_csv, censor=0, subset='multi'):
    df = pd.read_csv(data_csv)
    df.ID = [x.split('_')[0] for x in df.ID]
    cli = pd.read_csv(cli_csv)
    cli = cli.loc[:, ['id', 'event_overall_survival', 'overall_survival_in_days', 'event_distant_metastases',
                      'distant_metastases_in_days']]
    cli.columns = ['ID', 'mor', 'fu', 'dm', 'dmfu']
    if censor != 0:
        for i in range(len(cli)):
            if cli['fu'][i] > censor:
                cli.iloc[i, 1] = 0
                cli.iloc[i, 2] = censor
    df = pd.merge(df, cli, on='ID', how='inner')
    df = df.sort_values('original_shape_MeshVolume', ascending=False)
    n_lesions = df['ID'].value_counts()
    multifocal = n_lesions.index[n_lesions>1]
    multifocal_idx = [x in multifocal for x in df['ID']]
    primary_idx = ['GTV-1_1' in x for x in df['Mask']]
    df.ID = [x[2:] for x in df.ID]
    if subset == 'largest':
        df = df.drop_duplicates(subset='ID', keep='first')
        return df
    elif subset == 'multi':
        return df[multifocal_idx]
    elif subset == 'uni':
        return df[~np.array(multifocal_idx)]
    elif subset == 'multi_largest':
        return df[multifocal_idx].drop_duplicates(subset='ID', keep='first')
    elif subset == 'all':
        return df
    elif subset == 'primary':
        return df[primary_idx]
    else:
        print("Supported subsets are uni, multi and largest")

def feature_sets_hn(df, feature_class='original', normalize=True):
    # Keep original features only and discard Laplacian of Gaussian (Log) features and wavelet features
    idxs = df.columns.str.contains(r'sigma|wavelet|diagnostics')
    if feature_class == 'original':
        df = df.iloc[:, ~idxs]
    # Hard-coded feature columns, should be changed based on specific datasets
    X = df.iloc[:, 6: -2]
    # Two step normalization
    if normalize:
        # Log transformation with a correction term
        X = X - X.min() + np.median(X - X.min())
        X = np.log(X)
        # TODO: Neglog transformation
    X = (X - X.mean()) / X.std()
    # Column 0: Patient ID, column -2: follow up time and column -1, mortality were passed as bag labels
    Y = df.iloc[:, [1, -2, -1]]
    return X, Y

def create_lung_df(data_csv, cli_csv, censor=0, subset='multi'):
    df = pd.read_csv(data_csv)
    df.ID = [x.split('_')[0] for x in df.ID]
    cli = pd.read_csv(cli_csv)
    cli = cli.loc[:, ['PatientID', 'deadstatus.event', 'Survival.time']]
    cli.columns = ['ID', 'mor', 'fu']
    if censor != 0:
        for i in range(len(cli)):
            if cli['fu'][i] > censor:
                cli.iloc[i, 1] = 0
                cli.iloc[i, 2] = censor
    df = pd.merge(df, cli, on='ID', how='inner')
    df = df.sort_values('original_shape_MeshVolume', ascending=False)
    n_lesions = df['ID'].value_counts()
    multifocal = n_lesions.index[n_lesions>1]
    multifocal_idx = [x in multifocal for x in df['ID']]
    primary_idx = ['GTV-1_1' in x for x in df['Mask']]
    df.ID = [x[6:] for x in df.ID]
    if subset == 'largest':
        df = df.drop_duplicates(subset='ID', keep='first')
        return df
    elif subset == 'multi':
        return df[multifocal_idx]
    elif subset == 'uni':
        return df[~np.array(multifocal_idx)]
    elif subset == 'multi_largest':
        return df[multifocal_idx].drop_duplicates(subset='ID', keep='first')
    elif subset == 'all':
        return df
    elif subset == 'primary':
        return df[primary_idx]
    else:
        print("Supported subsets are uni, multi and largest")

def feature_sets_lung(df, feature_class='original', normalize=True):
    # Keep original features only and discard Laplacian of Gaussian (Log) features and wavelet features
    idxs = df.columns.str.contains(r'sigma|wavelet|diagnostics')
    if feature_class == 'original':
        df = df.iloc[:, ~idxs]
    # Hard-coded feature columns, should be changed based on specific datasets
    X = df.iloc[:, 6: -2]
    # Two step normalization
    if normalize:
        # Log transformation with a correction term
        X = X - X.min() + np.median(X - X.min())
        X = np.log(X)
    X = (X - X.mean()) / X.std()
    # Column 0: Patient ID, column -2: follow up time and column -1, mortality were passed as bag labels
    Y = df.iloc[:, [1, -2, -1]]
    return X, Y

if __name__ == '__main__':
    PET_csv = r'E:\PycharmProjects\AMINN_torch\data\HECKTOR\HECKTOR_test_PT.csv'
    CT_csv = r'E:\PycharmProjects\AMINN_torch\data\HECKTOR\HECKTOR_test_2_corrected.csv'
    cli_csv = r'E:\PycharmProjects\AMINN_torch\data\HECKTOR\hecktor2022_clinical_info_testing.csv'

    df_pet = pd.read_csv(PET_csv)
    df_ct = pd.read_csv(CT_csv)
    df_cli = pd.read_csv(cli_csv)
    df_both = pd.merge(df_ct, df_pet, 'inner', on='mpath')


    loc_list = df_both['diagnostics_Mask-original_CenterOfMassIndex_x'].to_list()
    loc_list = [ast.literal_eval(x) for x in loc_list]
    df_both['locx'] = [x[0] for x in loc_list]
    df_both['locy'] = [x[1] for x in loc_list]
    df_both['locz'] = [x[2] for x in loc_list]

    idxs = df_both.columns.str.contains(r'sigma|wavelet|diagnostics')
    df_both = df_both.iloc[:, ~idxs]

    df_both['ID_x'] = [x.split('_')[0] for x in df_both['ID_x']]
    df_both.rename(columns={'ID_x': 'PatientID'}, inplace=True)

    df_all = pd.merge(df_both, df_cli, on='PatientID')
    df_all_kept = df_all.drop(columns=['Task 1', 'Task 2', 'Tobacco', 'Alcohol', 'Performance_status', 'HPV status (0=-, 1=+)', 'Surgery', 'Chemotherapy'])
    df_all_kept.to_csv('HECKTOR_test_all_setting2.csv')
    print(1)
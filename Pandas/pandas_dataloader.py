"""
Script is about creating dataset for performing MIL type training for PANDAS
"""

from pathlib import Path
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd

class Pandas_Dataset(Dataset):
    """
    Loading graphs and defining labelset for pre/join learning
    """
    def __init__(self, data_df:pd.DataFrame, data_dir:str) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_df = data_df
        self.n_classes = self.data_df["isup_grade"].max() + 1

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        # int_id,image_id, data, label = self.data_set[index]
        label = int(self.data_df.iloc[index]["isup_grade"])
        target = torch.zeros(size=(1,self.n_classes-1))
        target[:,:label] = 1
        return {"id":self.data_df.iloc[index]["int_id"],
                "image":self.data_df.iloc[index]["data"],
                "label":target}

    def get_sample_weights(self):
        class_weights = self.data_df["isup_grade"].value_counts()
        self.data_df["sample"] = self.data_df["isup_grade"].apply(lambda x: 1/class_weights[x])
        return self.data_df["sample"].values
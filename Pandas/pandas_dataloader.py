#
# --------------------------------------------------------------------------------------------------------------------------
# Created on Fri dec 8 2023 at University of Toronto
#
# Author: Vishwesh Ramanathan
# Email: vishwesh.ramanathan@mail.utoronto.ca
# Description: This script is about creating dataset for pandas recov 
# Modifications (date, what was modified):
#   1.
# --------------------------------------------------------------------------------------------------------------------------
#

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
        # self.data_list = data_list
        # self.data_set = []
        # for i in range(len(self.data_df)):
        #     int_id = self.data_df.iloc[i]["int_id"]
        #     paths = self.data_df.iloc[i]["image_id"]
        #     data = torch.load(self.data_dir/f"{paths}_featvec.pt",map_location="cpu")
        #     label = self.data_df.iloc[i]["isup_grade"]
        #     self.data_set.append({"id":int_id,"paths":paths,"data":data,"label":label})
        # self.data_set = pd.DataFrame(self.data_set)
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
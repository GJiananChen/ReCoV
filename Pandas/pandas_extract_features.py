'''
Script for generating feature vectors per slide for MIL training on TCGA BRCA data
Sept 6th 2023: Modified to generate features on TCGA-BLCA
December 4th 2023: Modified to generate features on PANDAS
'''
import sys
import os
from collections import OrderedDict
from pathlib import Path
sys.path.append("/home/ramanav/Projects/TCGA_MIL/utils/Preprocess/")
sys.path.append("/aippmdata/trained_models/Martel_lab/pathology/SSL_CTransPath/")

import torch
import pandas as pd
import openslide
import numpy as np
from PIL import Image
from pathlib import Path
# import torch.utils.data as data
from tqdm import tqdm
import torchvision
import matplotlib
from matplotlib import pyplot as plt
from torch import nn, optim
import torch.nn.functional as F

from extract_patches import ExtractPatches
from get_features_CTransPath import model, trnsfrms_val


def load_model_weights(model, weights):
    model_dict = model.state_dict()
    weights = {k: v for k, v in weights.items() if k in model_dict}
    if weights == {}:
        print('No weight could be loaded..')
    model_dict.update(weights)
    model.load_state_dict(model_dict)

    return model

DEVICE = torch.device("cuda:0")

###################################################### LOAD THE MODELS ######################################################################################
# #Feature transform model
# FEATURE_MODEL_PATH = "/home/vramanathan/scratch/amgrp/simclr/resnet18_ozan/tenpercent_resnet18.ckpt"
# TRANSFORM_FV =  torchvision.transforms.Compose([
#                                             # transforms.Resize((224,224)),
#                                             torchvision.transforms.ToTensor(),
#                                             # torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                             #                     std=[0.229, 0.224, 0.225])
#                                             torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
#                                                                 std=[0.5, 0.5, 0.5])
#                                         ])

# model_fv = torchvision.models.__dict__['resnet18'](pretrained=False)
# state = torch.load(FEATURE_MODEL_PATH, map_location=DEVICE)
# state_dict = state['state_dict']
# for key in list(state_dict.keys()):
#     state_dict[key.replace('model.', '').replace('resnet.', '')] = state_dict.pop(key)

# model_fv = load_model_weights(model_fv, state_dict)
# model_fv.fc = torch.nn.Sequential()
# model_fv = model_fv.to(DEVICE)
# model_fv.eval()

model_fv = model.to(DEVICE)
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
TRANSFORM_FV = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean = mean, std = std)
    ]
)

#############################################################################################################################################################

# TILE_SIZE=256
TILE_SIZE=224
TILE_STRIDE_SIZE=1

# INPUT_DIR = list(Path("/aippmdata/public/TCGA-BRCA/images/").rglob("*.svs"))
# INPUT_DIR = list(Path("/aippmdata/public/TCGA-BLCA/images/").rglob("*.svs"))
# INPUT_DIR = list(Path("/aippmdata/public/TCGA-LUAD/images/").rglob("*.svs"))
INPUT_DIR = list(Path("/aippmdata/public/PANDAS/train_images/").glob("*.tiff"))
# OUTPUT_DIR = Path("/scratch/localdata0/vramanathan/TCGA_MIL_Patches_Ctrans_1MPP") #BRCA
# OUTPUT_DIR = Path("/scratch/localdata0/vramanathan/TCGA_MIL_Patches_Ctrans_1MPP_BLCA")
# OUTPUT_DIR = Path("/scratch/localdata0/vramanathan/TCGA_MIL_Patches_Ctrans_1MPP_LUAD")
OUTPUT_DIR = Path("/scratch/localdata0/vramanathan/PANDAS_MIL_Patches_Ctrans_1MPP")
Path.mkdir(OUTPUT_DIR,parents=True,exist_ok=True)
processed_files = [files.stem for files in OUTPUT_DIR.glob("*.pt")]

for paths in INPUT_DIR:  
    # slide_name = paths.stem.split(".")[0]
    slide_name = paths.stem
    print(f"Processing {slide_name}...")
    if slide_name in processed_files:
        print("Already processed...")
        continue
    # quality_decision = quality_check.loc[quality_check["Name"]==slide_name,"include"].item()

    patch_dataset = ExtractPatches(wsi_file = paths,
                                    patch_size = TILE_SIZE,
                                    # level_or_mpp = LWST_LEVEL_IDX,
                                    level_or_mpp=1.0,
                                    # foreground_threshold = 0.95,
                                    foreground_threshold = 0.5,
                                    patch_stride = TILE_STRIDE_SIZE,
                                    mask_threshold = 0.1,
                                    mask_kernelsize = 9,
                                    num_workers = 4,
                                    save_preview=False,
                                    save_mask=False,
                                    # output_dir="./",
                                    transform=TRANSFORM_FV,
                                    tta_transform=None)


    dataloader = torch.utils.data.DataLoader(patch_dataset, batch_size=512, num_workers=16)
    all_feats = []
    with torch.no_grad():
        for data in tqdm(dataloader,desc="Extracting and saving feature vectors"):
            # print(data)
            img = data[0].to(DEVICE)
            feats = model_fv(img)
            all_feats.extend(feats.cpu())
        all_feats = torch.stack(all_feats,dim=0)
    print("Extracted {} features".format(len(all_feats)))
    torch.save(all_feats,str(OUTPUT_DIR/f"{slide_name}_featvec.pt"))
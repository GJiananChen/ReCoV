import sys
import cv2
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings("ignore")

import skimage.io
from skimage import color
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import rescale
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from openslide import OpenSlide
from pathlib import Path

# Pretrained model from https://github.com/Xiyue-Wang/TransPath
sys.path.append("../../SSL_CTransPath/")
from get_features_CTransPath import model

def tile(path,mpp=1):
    scan = OpenSlide(path)
    img = skimage.io.MultiImage(path)[-1]
    level_dimensions = scan.level_dimensions
    image_array = np.asarray(scan.read_region((0, 0), len(level_dimensions)-1, level_dimensions[-1]).convert('RGB'))
    shape = img.shape
    
    #get mask from image
    threshold = 0.1
    lab = color.rgb2lab(image_array)
    mean = np.mean(lab[..., 1])
    lab = lab[..., 1] > (1 + threshold ) * mean
    mask = lab.astype(np.uint8)
    fill_mask_kernel_size=9
    mask = binary_fill_holes(mask)
    kernel = np.ones((fill_mask_kernel_size, fill_mask_kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    mask = (mask>0)*1
    downsample_factor = int(level_dimensions[0][0]/level_dimensions[-1][0])

    if mpp==1:
        sz_big = sz*2 
    elif mpp==0.5:
        sz_big = sz
    else:
        raise ValueError(f"Wrong MPP value of {mpp} select between 1 or 0.5")
    
    #Extract patches at regular stride
    lim0,lim1 = shape[0]-shape[0]%sz_big,shape[1]-shape[1]%sz_big 
    sz_mask = int(sz_big/downsample_factor)
    img = img[:lim0,:lim1,:]
    mask = mask[:int(lim0//downsample_factor),:int(lim1//downsample_factor)]
    img = img.reshape(img.shape[0]//sz_big,sz_big,img.shape[1]//sz_big,sz_big,3)
    mask = mask.reshape(mask.shape[0]//sz_mask,sz_mask,mask.shape[1]//sz_mask,sz_mask,1)
    img = img.transpose(0,2,1,3,4).reshape(-1,sz_big,sz_big,3)
    mask = mask.transpose(0,2,1,3,4).reshape(-1,sz_mask,sz_mask,1)
    idxs = np.where(mask.reshape(mask.shape[0],-1).sum(-1)/float(sz_mask*sz_mask)>=0.8)[0]
    assert mask.shape[0]==img.shape[0]
    img = img[idxs]
    #For 1MPP extraction
    if mpp==0.5:
        return img
    else:
        temp = []
        for i in range(len(img)):
            temp.append(rescale(img[i],0.5,channel_axis=-1))
        temp = np.stack(temp)
        return temp

class PandaDataset(Dataset):
    def __init__(self, path, test):
        self.path = path
        self.names = list(pd.read_csv(test).image_id)
        self.mean = torch.tensor([0.485, 0.456, 0.406])
        self.std = torch.tensor([0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        path = os.path.join(self.path,name+'.tiff')
        #Can make it faster at this stage will decide accordingly
        tiles = torch.Tensor(tile(path,MPP))
        tiles = (tiles - self.mean)/self.std
        return tiles.permute(0,3,1,2), name

def featextraction(tiles):
    dataloader = torch.utils.data.DataLoader(tiles, batch_size=256)
    all_feats = []
    with torch.no_grad():
        for data in dataloader:
            # print(data)
            img = data.cuda()
            feats = model_fv(img)
            all_feats.extend(feats)
        all_feats = torch.stack(all_feats,dim=0)
        print("Extracted {} features".format(len(all_feats)))
    return all_feats

DATA = '../data/PANDA/train_images'
TEST = '../data//PANDA/train.csv'
sz = 224
nworkers = 4
DEVICE = torch.device("cuda:0")
model_fv = model.to(DEVICE)
#Choose MPP between 1 or 0.5
MPP = 1
# MPP = 0.5
OUTPUT_DIR = Path(f'./data/PANDA/PANDA_MIL_Patches_Selfpipeline_{MPP}MPP')
Path.mkdir(OUTPUT_DIR,parents=True,exist_ok=True)
processed_files = [files.stem for files in OUTPUT_DIR.glob("*.pt")]

if os.path.exists(DATA):
    ds = PandaDataset(DATA,TEST)
    names,preds = [],[]
    with torch.no_grad():
        for idx in tqdm(range(len(ds))):
            name = ds[idx][1]
            if (name+'_featvec') in processed_files:
                print("Already processed...")
                continue
            tiles = ds[idx][0]
            all_feats = featextraction(tiles)
            torch.save(all_feats,str(OUTPUT_DIR/f"{name}_featvec.pt"))

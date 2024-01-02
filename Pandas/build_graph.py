import os

import torch
import numpy as np
from PIL import Image
from pathlib import Path
# import torch.utils.data as data
from tqdm import tqdm
import torchvision
from torch import nn, optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.data import Data as geomData
import scipy.sparse as sp
import nmslib

from dplabtools.slides import GenericSlide
from dplabtools.slides.patches import WholeImageGridPatches
from dplabtools.slides.processing import WSIMask
# from utils_graph.visualize_graph import read_slide, visualize_patches 

class Hnsw:
    def __init__(self, space='cosinesimil', index_params=None,
                 query_params=None, print_progress=True):
        self.space = space
        self.index_params = index_params
        self.query_params = query_params
        self.print_progress = print_progress

    def fit(self, X):
        index_params = self.index_params
        if index_params is None:
            index_params = {'M': 16, 'post': 0, 'efConstruction': 400}

        query_params = self.query_params
        if query_params is None:
            query_params = {'ef': 90}

        # this is the actual nmslib part, hopefully the syntax should
        # be pretty readable, the documentation also has a more verbiage
        # introduction: https://nmslib.github.io/nmslib/quickstart.html
        index = nmslib.init(space=self.space, method='hnsw')
        index.addDataPointBatch(X)
        index.createIndex(index_params, print_progress=self.print_progress)
        index.setQueryTimeParams(query_params)

        self.index_ = index
        self.index_params_ = index_params
        self.query_params_ = query_params
        return self

    def query(self, vector, topn, dist_norm):
        # the knnQuery returns indices and corresponding distance
        # we will return the normalized distance based on dist_norm
        indices, dist = self.index_.knnQuery(vector, k=topn)
        return indices, np.sqrt(dist)/dist_norm

def pt2graph(wsi_h5, radius=9, dist_threshold=3, patch_size=224*4):
    """
    Main function to form graph based on KNN calculated based on eucledian distance.
    We throw away nodes which are above dist_threshold*patch_size
    """
    coords, features = wsi_h5['coords'], wsi_h5['features']
    assert coords.shape[0] == features.shape[0]
    num_patches = coords.shape[0]
    
    model = Hnsw(space='l2')
    model.fit(coords)
    a = np.repeat(range(num_patches), radius-1)
    b = []
    for v_idx in range(num_patches):
        out = np.stack(model.query(coords[v_idx], topn=radius, dist_norm=patch_size),axis=1)[1:,]
        b.append(out)
    b = np.concatenate(b)
    b = np.concatenate((a[:,np.newaxis],b),axis=1)
    #Set a distance threshold at 2.8 patch distance
    edge_spatial = torch.Tensor(b[np.where(b[:,2]<=dist_threshold)[0],:2].T).type(torch.LongTensor)

    G = geomData(x = features,
                 edge_index = edge_spatial,
                 centroid = torch.Tensor(coords))
    return G

def find_best_level(scan, size):
    level_dimensions = scan.level_dimensions
    for i,levels in enumerate(level_dimensions[::-1]):
        #Select level with dimension around 1000 otherwise it becomes too big
        if levels[0]>size or levels[1]>size:
            break
    return len(level_dimensions)-1-i

def get_coordinates(wsi_file):
    slide = GenericSlide(wsi_file)
    level = find_best_level(scan=slide,size=1000)
    mask = WSIMask(wsi_file, level=level, mode="lab", threshold=0.1, fill_mask_kernel_size=9)

    grid_patches = WholeImageGridPatches(wsi_file = wsi_file,
                                    mask_data = mask.array,
                                    patch_size = TILE_SIZE,
                                    level_or_mpp = 1.0,
                                    foreground_threshold=0.5,
                                    patch_stride = TILE_STRIDE_SIZE,
                                    overlap_threshold = 1.0,
                                    weak_label = "label")
    coordinates = np.array([coords[0] for coords in grid_patches.patch_data])
    return coordinates
    
if __name__=="__main__":
    TILE_SIZE=224
    TILE_STRIDE_SIZE=1
    #graph building hyperparameters
    DIST_THRESH = 8
    # DIST_THRESH = 10
    # DIST_THRESH = 20

    # INPUT_DIR = list(Path("/aippmdata/public/TCGA-BRCA/images/").rglob("*.svs"))
    # OUTPUT_DIR = Path("/localdisk3/ramanav/TCGA_processed/TCGA_MIL_TILgraph/knn_graph_nosample_pe_crds")
    INPUT_DIR = Path("/aippmdata/public/PANDAS/train_images/").glob("*.tiff")
    FEAT_DIR = Path("/localdisk3/ramanav/TCGA_processed/PANDAS_MIL_Patches_Ctrans_1MPP")
    OUTPUT_DIR = Path("/localdisk3/ramanav/TCGA_processed/pandas_graph")
    processed_files = [files.stem for files in OUTPUT_DIR.glob("*.pt")]
    Path.mkdir(OUTPUT_DIR,parents=True,exist_ok=True)
    
    for paths in INPUT_DIR:
        slide_name = paths.stem
        slide = GenericSlide(paths)
        spacing = slide.mpp_data[0]
        ih, iw = slide.level_dimensions[0]
        print(f"Processing {slide_name}")
        patch_size = (1/spacing)*TILE_SIZE
        if slide_name in processed_files:
            print("Already processed...")
            continue
        data = torch.load(FEAT_DIR/f"{slide_name}_featvec.pt",map_location="cpu")
        coords = get_coordinates(paths)
        feats = data
        G = pt2graph({"coords":coords,"features":feats},dist_threshold=DIST_THRESH,patch_size=patch_size)
        torch.save(G, str(OUTPUT_DIR/f"{slide_name}_graph.pt"))
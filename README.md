# Detecting noisy labels with Repeated Cross-Validations - ReCoV
Code and data for the manuscript: **Detecting noisy labels with repeated cross-validations**

## Description

This repository contains the code and data of the paper "Detecting noisy labels with repeated
cross-validations", which is accepted at MICCAI 2024 for publication.

In this work we propose a novel algorithm for identifying cases with noisy labels within a given dataset.
We found that noisy cases consistently lead to worse performance for the validation fold in n-fold cross validations. By pinpointing the examples that more frequently contribute to inferior cross-validation results, our methods, ReCoV and fastReCoV, effectively identifies the noisy samples within the dataset.
 


## Key Features
- State-of-the-Art Performance: fastReCoV outperforms existing methods for noisy label detection in popular comptuer vision and medical imaging datasets
- Plug-and-play for most supervised learning tasks and network structures, with potential applications beyond computer vision. 
- Does not require prior knowledge of the percentage of noisy examples.
- Efficient when using embeddings (extracted with pre-trained models) as inputs


## Methodology
The methodlogy consists of two algorithms:-
1. ReCoV, the original algorithm that is grounded in mathematical foundations. Recommended for tabular datasets and embeddings. 
2. FastReCoV, a computationally efficient variant that offers a slightly reduced performance but are significantly faster. Recommended for deep learning tasks with large datasets.
An overview of the methodology and its results are shown below

<img src="https://github.com/GJiananChen/ReCoV/blob/master/images/recov.JPG" align="center" width="880" ><figcaption>Fig.1 - Pseudocode for ReCoV</figcaption></a>

<img src="https://github.com/GJiananChen/ReCoV/blob/master/images/fastrecov.JPG" align="center" width="880" ><figcaption>Fig.2 - Pseudocode for FastReCoV</figcaption></a> 

## Getting Started

### Dependencies

```
opencv
pytorch-gpu
wandb
openslide
scikit-learn
scipy
scikit-image
warmup-scheduler
nystrom-attention
```
### Datasets

The project is applied to 4 datasets:-  
1. Mushroom (`./mushroom`) - https://www.kaggle.com/datasets/uciml/mushroom-classification
2. Hecktor (`./HECKTOR`) - https://hecktor.grand-challenge.org/
3. CIFAR-10N (`./cifar10n`) - http://noisylabels.com/
4. PANDA (`./PANDA`) - https://www.kaggle.com/competitions/prostate-cancer-grade-assessment


### Reproducing our results

#### Mushroom dataset
The dataset can be downloaded from the given link. The two algorithms can be run directly using ```python mushroom_[recov/fastrecov].py```
#### CIFAR10N dataset
Before running the model on the dataset, for the individual images, features are to be extracted. This can be done via ```python cifar_featextract.py```. After this step, fastRecov can be run using ```python cifar_fastrecov.py```
#### HECKTOR dataset
Before running the model, the radiomics features are extracted using ```python HECKTOR_extraction.py``` After this step, both recov and fastrecov can be run using ```python HECKTOR_[recov/fastrecov].py```. The identified labels can be evaluated using ```python hecktor_evaluate.py```
#### PANDA dataset
Before running the model, the feautres are to be extracted using ```python featureextraction.py```. After this step, fastRecov can be run using ```python panda_fastrecov.py```. Since the test set is hosted in kaggle, users can save the fastrecov noise cleaned model using ```python test_fastrecov.py```, and evalute on kaggle using ```panda-recov-submission.ipynb``` notebook.
## Results

<img src="https://github.com/GJiananChen/ReCoV/blob/master/images/cifar10n_results.png" align="center" width="880" ><figcaption>Fig.3 - Results for CIFAR10N dataset</figcaption></a>
For more results, please refer to our paper.
More results to be added after the camera ready submission.

### Applying ReCoV to your dataset
To apply ReCoV to your dataset, you can follow the steps below:-
1. Prepare your dataset by converting them into a tabular dataset (-omics) or extracting features from your cases with a pre-trained model e.g. .
2. Run the fastReCoV algorithm by following the pseudo code or example scripts.

## Contact
You can reach the authors by raising an issue in this repo or
 email them at chenjn2010@gmail.com/vishweshramanathan@mail.utoronto.ca/a.martel@utoronto.ca

## Cite
```
@article{chen2023cross,
  title={Cross-Validation Is All You Need: A Statistical Approach To Label Noise Estimation},
  author={Chen, Jianan and Martel, Anne},
  journal={arXiv preprint arXiv:2306.13990},
  year={2023}
}
```
To be updated with new citations after publication.

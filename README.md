# Detecting noisy labels with Repeated Cross-Validations - ReCOV
Code and data for the manuscript: **Detecting noisy labels with repeated cross-validations**

## Description

This project is about label noise detection problem from any dataset, especially targeting medical datasets. Two algorithms namely ReCoV and its faster alternative, fastReCoV are proposed, based on the idea that fluctuations in cross validation performance are caused by label noise. The algorithms achieve state of the art label noise detection performance in a wide range of modalities, models and tasks. 

## Methodology
The methodlogy consists of two algorithms:-
1. ReCoV
2. FastReCoV
An overview of the methodology and its results are shown below

<img src="https://github.com/GJiananChen/ReCoV/blob/sample/images/recov.png" align="center" width="880" ><figcaption>Fig.1 - Pseudocode for ReCoV</figcaption></a>

<img src="https://github.com/GJiananChen/ReCoV/blob/sample/images/fastrecov.png" align="center" width="880" ><figcaption>Fig.2 - Pseudocode for FastReCoV</figcaption></a> 

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
4. PANDAS (`./Pandas`) - https://www.kaggle.com/competitions/prostate-cancer-grade-assessment


### Preparing dataset and Running ReCoV
#### Mushroom dataset
The dataset can be downloaded from the given link. The two algorithms can be run directly using ```python mushroom_[recov/fastrecov].py```
#### CIFAR10N dataset
Before running the model on the dataset, for the individual images, features are to be extracted. This can be done via ```python cifar_featextract.py```. After this step, fastRecov can be run using ```python cifar_fastrecov.py```
#### HECKTOR dataset
Before running the model, the radiomics features are extracted using .... After this step, both recov and fastrecov can be run using ```python HECKTOR_[recov/fastrecov].py```. The identified labels can be evaluated using ```python hecktor_evaluate.py```
#### PANDAS dataset
Before running the model, the feautres are to be extracted using ```python featureextraction.py```. After this step, fastRecov can be run using ```python pandas_fastrecov.py```. Since the test set is hosted in kaggle, users can save the fastrecov noise cleaned model using ```python test_fastrecov.py```, and evalute on kaggle using ```pandas-recov-submission.ipynb``` notebook.
## Results
For more results, please refer to our paper  

<img src="https://github.com/GJiananChen/ReCoV/blob/sample/images/cifar10n_results.png" align="center" width="880" ><figcaption>Fig.3 - Results for CIFAR10N dataset</figcaption></a>

## Contact
If you want to contact, you can reach the authors by raising an issue or
 email at vishweshramanathan@mail.utoronto.ca/chenjn2010@gmail.com

## Cite
```
```

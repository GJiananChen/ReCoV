import torch
from torchvision import datasets
from torchvision import transforms
from torchvision.models import vit_b_16
import tqdm
import h5py

#Load Model
# CHANGE THE MODEL TYPE FOR DIFFERENT FEATURE VECTORS
FEAT_MODE = "dinov2"
# FEAT_MODE = "imagenet"
BATCH_SIZE = 4

#DINO v2
if FEAT_MODE=="dinov2":
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
elif FEAT_MODE=="imagenet":
    model = vit_b_16(weights="DEFAULT")
    model.heads = torch.nn.Sequential()
else:
    raise NotImplementedError
model = model.cuda()
model.eval()
print(model)

img_trans = transforms.Compose([
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

#Load dataset
trainset = datasets.CIFAR10(root='../data', train=True, download=True, transform=img_trans)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

testset = datasets.CIFAR10(root='../data', train=False, download=True, transform=img_trans)
test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

#Extract features and labels for training
with torch.no_grad():
    train_feats = []
    train_labels = []
    for img, label in tqdm.tqdm(train_loader):
        img= img.cuda()

        feats = model(img)

        train_feats.extend(feats.cpu().numpy())
        train_labels.extend(label.numpy())
        
#Extract features and labels for training
with torch.no_grad():
    test_feats = []
    test_labels = []
    for img, label in tqdm.tqdm(test_loader):
        img= img.cuda()

        feats = model(img)

        test_feats.extend(feats.cpu().numpy())
        test_labels.extend(label.numpy())

#Create dataset
print(f"Training dataset (Num of feats, Num of labels): {len(train_feats)}, {len(train_labels)}")
print(f"Test datasets (Num of feats, Num of labels): {len(test_feats)}, {len(test_labels)}")

with h5py.File(f'cifar_feats_{FEAT_MODE}.h5', 'w') as f:
    f.create_dataset('train_feats', data=train_feats)
    f.create_dataset('train_labels', data=train_labels)
    f.create_dataset('test_feats', data=test_feats)
    f.create_dataset('test_labels', data=test_labels)

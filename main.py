import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torchvision
from utils.transform import VideoTransform
from utils.dataset import UCF101Dataset
from train import train_model

if __name__ == '__main__':
    train_dataset = UCF101Dataset(
    root='../data/UCF-101',
    annotation_path='../data/UCF_annotations',
    frames_per_clip=16,
    train=True,
    step_between_clips=4,
    num_workers=48,
    transform=VideoTransform(resize=(224, 224), phase='train')
    )

    val_dataset = valdataset = UCF101Dataset(
    root='../data/UCF-101',
    annotation_path='./data/UCF_annotations',
    frames_per_clip=16,
    step_between_clips=4,
    train=False,
    num_workers=48,
    transform=VideoTransform(resize=(224, 224), phase='val')
    )

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)
    
    mydataloadersdict = {'train' : train_dataloader, 'val' : val_dataloader}

    from model.eco_lite import ECO_Lite
    mymodel = ECO_Lite()
    myloss = nn.CrossEntropyLoss()
    myoptimizer = torch.optim.Adam(mymodel.parameters(), lr=0.001, betas=(0.9, 0.999))

    train_model(mymodel, mydataloadersdict, myloss, myoptimizer, 5)
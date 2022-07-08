import torch
import sys
import numpy as np
from dataset import make_datapath_list, VideoTransform, get_label_id_dictionary, VideoDataset
from model.eco_lite import ECO_Lite
from train import train_model

import argparse
import time
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=5, help="number of classes")
    parser.add_argument("--train_root_path", type=str, default='../data/UCF-101_C10/train/', help="train files path")
    parser.add_argument("--val_root_path", type=str, default='../data/UCF-101_C10/val/', help="validation files path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help = "Optional path to checkpoint model")
    parser.add_argument("--classid", type=str, default="./classid.txt", help="class & id config")
    parser.add_argument("--save_interval", type=int, default=4, help="save interval")

    opt = parser.parse_args()
    print(opt)
    
    train_path_list = make_datapath_list(opt.train_root_path)
    val_path_list = make_datapath_list(opt.val_root_path)
    resize, crop_size = 224, 224
    mean, std = [103, 117, 123], [1,1,1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    id_label_dict, label_id_dict = get_label_id_dictionary(class_id_path=opt.classid)

    train_dataset = VideoDataset(
        train_path_list, label_id_dict, num_segments=16, phase='train', transform=video_transform
    )
    val_dataset = VideoDataset(
        val_path_list, label_id_dict, num_segments=16, phase='val', transform=video_transform
    )

    batch_size = opt.batch_size
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    whole_dataloader = {
        'train' : train_dataloader,
        'val' : val_dataloader
    }

    my_model = ECO_Lite(num_of_classes=opt.classes)
    if opt.checkpoint is not None:
        my_model = torch.load(opt.checkpoint)
    my_loss = torch.nn.CrossEntropyLoss()
    my_opt = torch.optim.Adam(my_model.parameters(),lr=1e-3, betas=(0.9, 0.999))

    EPOCH = opt.epochs
    train_model(my_model, whole_dataloader, my_loss, my_opt, EPOCH, opt.save_interval)
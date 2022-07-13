import torch
import sys
import os
import numpy as np
from dataset import make_datapath_list, VideoTransform, get_label_id_dictionary, VideoDataset
from model.eco_lite import ECO_Lite


import argparse
import time
import datetime

import torch
from tqdm import tqdm
import time
import copy

def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs -1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, _, labels, _ in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders_dict[phase].dataset)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
                if not os.path.exists('./weights'):
                    os.mkdir('./weights')
                name = 'best_' + str(epoch) + '.pth'
                save_path = os.path.join('./weights', name)
                torch.save(model.state_dict(), save_path)
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')   

    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=101, help="number of classes")
    parser.add_argument("--data_root_path", type=str, default='../data/UCF-101/train', help="train files path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("--epochs", type=int, default=10, help="number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help = "Optional path to checkpoint model")
    parser.add_argument("--savepath", type=str, default='weights/best.pth', help="Save path config")
    parser.add_argument("--classid", type=str, default="./classid.txt", help="class & id config")

    opt = parser.parse_args()
    print(opt)
    
    train_path_list = make_datapath_list(opt.data_root_path)
    resize, crop_size = 224, 224
    mean, std = [103, 117, 123], [1,1,1]
    video_transform = VideoTransform(resize, crop_size, mean, std)

    id_label_dict, label_id_dict = get_label_id_dictionary(class_id_path=opt.classid)

    wholedataset = VideoDataset(
        train_path_list, label_id_dict, num_segments=16, phase='train', transform=video_transform
    )

    train_size = int(0.8 * len(wholedataset))
    test_size = len(wholedataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(wholedataset, [train_size, test_size])

    batch_size = opt.batch_size
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    whole_dataloader = {
        'train' : train_dataloader,
        'val' : val_dataloader
    }

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    my_model = ECO_Lite(num_of_classes=opt.classes)
    if opt.checkpoint is not None:
        my_model = torch.load(opt.checkpoint)
    my_loss = torch.nn.CrossEntropyLoss()
    # my_opt = torch.optim.Adam(my_model.parameters(),lr=1e-3, betas=(0.9, 0.999))
    my_opt = torch.optim.SGD(my_model.parameters(), lr=1e-3, momentum=0.9)

    my_model.to(device)
    result = train_model(my_model, whole_dataloader, my_loss, my_opt, opt.epochs)
    torch.save(result.state_dict(), opt.savepath)
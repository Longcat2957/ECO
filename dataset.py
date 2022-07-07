import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from PIL import Image

def make_datapath_list(root_path='../data/UCF-101/'):
    """
    return folder_list
    root_path : str, root_path of data ('~/data/UCF-101')
    """
    video_list = []
    class_list = os.listdir(path=root_path)

    for class_list_i in (class_list):
        class_path = os.path.join(root_path, class_list_i)

        for file_name in os.listdir(class_path):
            name, ext = os.path.splitext(file_name) # ignore formats(*.avi)
            video_img_dir = os.path.join(class_path, name)

            video_list.append(video_img_dir)

    return video_list

class VideoTransform():
    """
    Video to images, Group Transform
    """

    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = {
            'train' : T.Compose([
                GroupDataAugmentation(int(resize)),
                GroupToTensor(),
                GroupImgNormalize(mean, std),
                Stack()
            ]),
            'val' : T.Compose([
                GroupResize(int(resize)),
                GroupCenterCrop(crop_size),
                GroupToTensor(),
                GroupImgNormalize(mean, std),
                Stack()
            ])
        }
    
    def __call__(self, img_group, phase):
        return self.data_transform[phase](img_group)

# preprocess classes

class GroupDataAugmentation():
    def __init__(self, resize):
        self.set = T.Compose([
            T.RandomResizedCrop(
                resize, scale=(0.25, 1.0)
            ),
            T.RandomHorizontalFlip()
        ])
    def __call__(self, img_group):
        return [self.set(img) for img in img_group]

class GroupResize():
    def __init__(self, resize, interpolation=Image.BILINEAR):
        self.rescaler = T.Resize(resize, interpolation)

    def __call__(self, img_group):
        return [self.rescaler(img) for img in img_group]

class GroupCenterCrop():
    def __init__(self, crop_size):
        self.ccrop = T.CenterCrop(crop_size)
    
    def __call__(self, img_group):
        return [self.ccrop(img) for img in img_group]

class GroupToTensor():
    def __init__(self):
        self.to_tensor = T.ToTensor()

    def __call__(self, img_group):
        return [self.to_tensor(img)*255 for img in img_group]

class GroupImgNormalize():
    def __init__(self, mean, std):
        self.normalize = torchvision.transforms.Normalize(mean, std)

    def __call__(self, img_group):
        return [self.normalize(img) for img in img_group]

class Stack():

    def __call__(self, img_group):
        ret = torch.cat([(x.flip(dims=[0])).unsqueeze(dim=0) for x in img_group], dim=0)
        return ret

def get_label_id_dictionary(class_id_path='classid.txt'):
    empty_dict = {}
    empty_dict_2 = {}
    with open(class_id_path, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            a, b = line.strip().split()
            empty_dict[int(a)] = b
            empty_dict_2[b] = int(a)
    return empty_dict, empty_dict_2

class VideoDataset(Dataset):
    def __init__(self, video_list, label_id_dict, num_segments, phase, transform, img_tmpl='image_{:05d}.jpg'):
        self.video_list = video_list
        self.label_id_dict = label_id_dict
        self.num_segments = num_segments # 동영상 분할 방법
        self.phase = phase # 'train' or 'val'
        self.transform = transform
        self.img_tmpl = img_tmpl
    
    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, index):

        imgs_transformed, label, label_id, dir_path = self.pull_item(index)
        return imgs_transformed, label, label_id, dir_path

    def pull_item(self, index):

        dir_path = self.video_list[index]
        indices = self._get_indices(dir_path)
        img_group = self._load_imgs(
            dir_path, self.img_tmpl, indices
        )

        label = (dir_path.split('/')[4].split('/')[0])
        label_id = self.label_id_dict[label]

        imgs_transformed = self.transform(img_group, phase=self.phase)
        return imgs_transformed, label, label_id, dir_path

    def _load_imgs(self, dir_path, img_tmpl, indices):
        img_group = []

        for idx in indices:
            file_path = os.path.join(dir_path, img_tmpl.format(idx))
            img = Image.open(file_path).convert('RGB')
            img_group.append(img)
        return img_group

    def _get_indices(self, dir_path):
        file_list = os.listdir(path=dir_path)
        num_frames = len(file_list)

        # 동영상 간격 구하기
        tick = (num_frames) / float(self.num_segments)
        indices = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)]) + 1

        return indices
if __name__ == '__main__':
    datapath = '../data/UCF-101'
    video_list = make_datapath_list(datapath)
    resize, crop_size = 224, 224
    mean, std = [104, 117, 123], [1, 1, 1]
    video_transform = VideoTransform(resize, crop_size, mean, std)
    
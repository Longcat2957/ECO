import json
import os
import pathlib
import torch
import numpy as np
import scipy
from torch.utils.data import Dataset

_COCO_JOINTS = [
    'Nose',
    'Left Eye',
    'Right Eye',
    'Left Ear',
    'Right Ear',
    'Left Shoulder',
    'Right Shoulder',
    'Left Elbow',
    'Right Elbow',
    'Left Wrist',
    'Right Wrist',
    'Left Hip',
    'Right Hip',
    'Left Knee',
    'Right Knee',
    'Left Ankle',
    'Right Ankle'
]


def make_annotation_list(root_path:str):
    annotations_list = []
    for (root, dirs, files) in os.walk(root_path):
        dir_path = os.path.join(root_path, root)
        for file in files:
            name, ext = os.path.splitext(file)
            if ext == '.json':
                file_path = os.path.join(dir_path, file)
                annotations_list.append(file_path)
    print(f'annotations no. {len(annotations_list)}')
    return annotations_list


class aihub_annotations(object):
    def __init__(self):
        self.frame = None
        self.type = None
        self.type_info = None
        self.item = None

    def push(self, annotation_path):
        with open(annotation_path, 'r') as f:
            data = json.load(f)
        self.frame = data['frames']
        self.type = data['type']
        self.type_info = data['type_info']

        item = []
        for v in range(1, 6):
            views = []
            for i in range(16):
                joints_dict = self.frame[i][f'view{v}']['pts']
                j = []
                for coco in _COCO_JOINTS:
                    x, y = joints_dict[coco]['x'], joints_dict[coco]['y']
                    j.append((x, y))
                views.append(j)
            item.append(views)

        self.item = item    # len(self.item) = 5

    def pull(self):
        if self.item is not None:
            return self.item
        else:
            raise KeyError
        

if __name__ == '__main__':
    train_annotations_list = make_annotation_list('/home/vlsimin95/Downloads/피트니스 자세 이미지/Training')
    one_of_annotations = train_annotations_list[0]
    with open(one_of_annotations, 'r') as f:
        data = json.load(f)
    
    obj = aihub_annotations()
    obj.push(one_of_annotations)
    print(obj.type)
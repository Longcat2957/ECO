import json
import os
import pathlib
import torch
import numpy as np
import scipy
import seaborn
from torch.utils.data import Dataset
import cv2
import pickle as pkl

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

_WO_DICT = {
    '스탠딩 사이드 크런치' : 0,
    '스탠딩 니업' : 1,
    '버피 테스트' : 2,
    '스텝 포워드 다이나믹 런지' : 3,
    '스텝 백워드 다이나믹 런지' : 4,
    '사이드 런지' : 5,
    '크로스 런지' : 6,
    '굿모닝' : 7,
    '라잉 레그 레이즈' : 8,
    '크런치' : 9,
    '바이시클 크런치' : 10,
    '시저크로스' : 11,
    '힙쓰러스트' : 12,
    '플랭크' : 13,
    '푸시업' : 14,
    '니푸쉬업' : 15,
    'Y - Exercise' : 16
}

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
    '''
    COCO annotations 기반
    '''
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
            for i in range(len(self.frame)):
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
            final = []
            for x in self.item:
                try:
                    final.append((_WO_DICT[self.type_info['exercise']], x))
                except:
                    continue
            return final
        else:
            raise KeyError


if __name__ == '__main__':
    train_annotations_list = make_annotation_list('/home/vlsimin95/Downloads/피트니스 자세 이미지/Validation')
    save = []
    for annotation_path in train_annotations_list:
        obj = aihub_annotations()
        obj.push(annotation_path)
        out = obj.pull()
        for o in out:
            save.append(o)
    
    with open('val.pkl', 'wb') as f:
        pkl.dump(save, f, pkl.HIGHEST_PROTOCOL)
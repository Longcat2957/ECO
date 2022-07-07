import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import UCF101
from torchvision.transforms import ToTensor

class UCF101Dataset(UCF101):
    '''
    사용자 정의 Dataset 클래스는 반드시 3개 함수를 구현해야 합니다. : __init__, __len__, and __getitem__.
    '''
    def __getitem__(self, idx):
        video, audio, info, video_idx = self.video_clips.get_clip(idx)
        label = self.samples[self.indices[video_idx]][1]

        if self.transform is not None:
            video = self.transform(video)

        return video, label
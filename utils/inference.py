import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from avi2jpg import avi2jpg
import torchvision
import torchvision.transforms as T


# input파일이 동영상인지 이미지로 이루어진 디렉토리인지 확인하는 함수
# 동영상을 이미지로 다시 분할하는 함수 avi2jpg 클래스 호출을 통해 구현

class VideoTransform_i():
    """
    for video_preproc
    """
    def __init__(self, resize, crop_size, mean, std):
        self.data_transform = T.Compose([
                GroupResize(int(resize)),
                GroupCenterCrop(crop_size),
                GroupToTensor(),
                GroupImgNormalize(mean, std),
                Stack()])
    
    def __call__(self, img_group):
        return self.data_transform(img_group)

# preprocess classes
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


class preprocECO(object):
    def __init__(self, img_tmpl='image_{:05d}.jpg'):
        self.num_segments = 16
        self.video_converter = avi2jpg()
        self.video_transformer = VideoTransform_i(224, 224, [104, 117, 123], [1, 1, 1])
        self.video_format_list = ['.avi', '.mp4']
        self.img_tmpl = img_tmpl

    def __call__(self, input):
        input_type = self._interprete_type(input)
        if input_type:
            if input_type == 'video':
                imgs_group_path = self._video_preproc(input)
            elif input_type == 'dir':
                imgs_group_path = input
            
            indices = self._get_indices(imgs_group_path) # [  5  13  21  29  37  46  54  62  70  78  86  95 103 111 119 127]
            img_group = self._load_imgs(imgs_group_path, self.img_tmpl, indices)

            return self.video_transformer(img_group)
        
        else:
            print("ERROR[1]")
    
    def _interprete_type(self, input):
        if os.path.isfile(input):
            if os.path.splitext(input)[1] in self.video_format_list:
                return 'video'
            else:
                return False
        elif os.path.isdir(input):
            return 'dir'
    
    def _video_preproc(self, video):
        name, format = os.path.splitext(video)
        name += '_images'
        if not os.path.exists(name):
            os.mkdir(name)
        
        imgs_path = os.path.join('.', name)
        self.video_converter.convert(video, imgs_path)
        return imgs_path

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
    # 비디오 추론 전처리를 위한 클래스 debug
    #test_video_file_path = 'test.avi'
    test_img_group_dir_path = './test_dir'

    test_preprocECO = preprocECO()
    #print(test_preprocECO._interprete_type(test_video_file_path))
    print(test_preprocECO._interprete_type(test_img_group_dir_path))
    #print(test_preprocECO(test_video_file_path))
    output = test_preprocECO(test_img_group_dir_path)
    print(output.shape)
    

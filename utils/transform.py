from typing import Tuple
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import pytorchvideo.transforms as Tv

class VideoTransform(object):

    def __init__(self, phase:str, resize:Tuple):
        self.resize = resize
        self.phase = phase
        self.data_transform = {
            'train' : T.Compose(
                [
                    GroupTranspose(),
                    GroupTypeEncoder(),
                    GroupResizer(self.resize),
                    GroupRandomize(),
                    GroupNormalize()
                ]
            ),
            'val' : T.Compose(
                [
                    GroupTranspose(),
                    GroupTypeEncoder(),
                    GroupResizer(self.resize),
                    GroupNormalize()

                ]
            )
        }
    
    def __call__(self, img_group):
        return self.data_transform[self.phase](img_group)

class GroupTranspose():
    '''
    input : torch.Size([B, 16, 240, 320, 3])
    ouput : torch.size([B, 16, 3, 240, 320])
    '''
    def __init__(self):
        pass

    def __call__(self, input):
        output = input.transpose(1, 3)
        output = output.transpose(2, 3)
        return output


class GroupResizer():
    def __init__(self, resize:Tuple):
        self.resize = resize
        self.rescaler = T.Resize(resize)

    def __call__(self, batched_vid):
        time, channel, width, height = batched_vid.shape
        out = batched_vid.reshape(time * channel, width, height)
        out = self.rescaler(out)
        out = out.reshape(time, channel, self.resize[0], self.resize[1])
        return out


class GroupRandomize():

    def __init__(self):
        self.random = Tv.RandAugment()

    def __call__(self, batched_vid):
        batched_vid = self.random(batched_vid)
        return batched_vid


class GroupNormalize():

    def __init__(self):
        self.normalizer = Tv.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))

    def __call__(self, batched_vid):
        batched_vid = batched_vid.transpose(0, 1)
        batched_vid = self.normalizer(batched_vid)
        batched_vid = batched_vid.transpose(0, 1)
        return batched_vid

class GroupTypeEncoder():

    def __init__(self):
        self.encoder = Tv.ConvertUint8ToFloat()
    def __call__(self, batched_vid):
        return self.encoder(batched_vid)

if __name__ == '__main__':
    # Chaeck GroupTranspose()
    gt = GroupTranspose()
    gt_input = torch.randn(16, 240, 320, 3)
    gt_out = gt(gt_input)
    print(gt_out.shape)

    # Check_VideoTransform
    input = torch.randn(16, 240, 320, 3)
    videotransform = VideoTransform(resize=(224, 224), phase='train')
    out_t = videotransform(input)
    print(out_t.shape)
    out_v = videotransform(input)
    print(out_v.shape)
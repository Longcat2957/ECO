import torch
import torch.nn as nn

from eco3d import Resnet_3D_3, Resnet_3D_4, Resnet_3D_5

class PoseNeck(nn.Module):
    def __init__(self, joints):
        super(PoseNeck, self).__init__()
        self.conv3d1 = nn.Conv3d(
            joints,
            96,
            kernel_size=(3, 3, 3),
            stride = (2, 2, 2),
            padding = (1, 1, 1)
        )
        self.bn = nn.BatchNorm3d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True 
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv3d2 = nn.Conv3d(
            96,
            96,
            kernel_size = (3, 3, 3),
            stride = (1, 1, 1),
            padding = (1, 1, 1)
        )
        self.bn2 = nn.BatchNorm3d(
            96, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True 
        )
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv3d1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv3d2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class PoseECO(nn.Module):
    def __init__(self, joints:int, actions:int):
        super(PoseECO, self).__init__()

        # Pose Neck 모듈
        self.neck = PoseNeck(joints)

        # 3D Net 모듈
        self.res3d3 = Resnet_3D_3()
        self.res3d4 = Resnet_3D_4()
        self.res3d5 = Resnet_3D_5()
        
        # Global Average Pooling
        self.global_pool = nn.AvgPool3d(
            kernel_size=(2, 7, 7), stride=1, padding=0
        )

        # 클래스 분류의 전결합층
        self.fc_final = nn.Linear(in_features=512, out_features=actions, bias=True)

    def forward(self, x):

        x = self.neck(x)
        x = self.res3d3(x)
        x = self.res3d4(x)
        x = self.res3d5(x)
        x = self.global_pool(x)
        x = x.view(x.size()[0], x.size()[1])
        x = self.fc_final(x)
        return x

if __name__ == '__main__':
    dummy_heatmap = torch.randn(1, 17, 16, 56, 56)

    neck = PoseECO(17, 17)
    output = neck(dummy_heatmap)
    print(f'output_shape = {output.shape}')

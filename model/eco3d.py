import torch
import torch.nn as nn

class Resnet_3D_3(nn.Module):
    def __init__(self):
        super(Resnet_3D_3, self).__init__()

        self.res3a_2 = nn.Conv3d(
            96, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding= (1, 1, 1)
        )
        self.res3a_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res3a_relu = nn.ReLU(inplace=True)

        self.res3b_1 = nn.Conv3d(
            128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.res3b_1_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res3b_1_relu = nn.ReLU(inplace=True)
        self.res3b_2 = nn.Conv3d(
            128, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.res3b_bn = nn.BatchNorm3d(
            128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res3b_relu = nn.ReLU(inplace = True)

    def forward(self, x):
        
        residual = self.res3a_2(x)
        out = self.res3a_bn(residual)
        out = self.res3a_relu(out)

        out = self.res3b_1(out)
        out = self.res3b_1_bn(out)
        out = self.res3b_relu(out)
        out = self.res3b_2(out)

        out += residual

        out = self.res3b_bn(out)
        out = self.res3b_relu(out)

        return out

class Resnet_3D_4(nn.Module):
    def __init__(self):
        super(Resnet_3D_4, self).__init__()

        self.res4a_1 = nn.Conv3d(
            128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.res4a_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res4a_1_relu = nn.ReLU(inplace=True)
        
        self.res4a_2 = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.res4a_down = nn.Conv3d(
            128, 256, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1) 
        )
        self.res4a_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res4a_relu = nn.ReLU(inplace=True)

        self.res4b_1 = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.res4b_1_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res4b_1_relu = nn.ReLU(inplace=True)
        self.res4b_2 = nn.Conv3d(
            256, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

        self.res4b_bn = nn.BatchNorm3d(
            256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res4b_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.res4a_down(x)
        
        out = self.res4a_1(x)
        out = self.res4a_1_bn(out)
        out = self.res4a_1_relu(out)

        out = self.res4a_2(out)

        out += residual

        residual2 = out

        out = self.res4a_bn(out)
        out = self.res4a_relu(out)

        out = self.res4b_1(out)

        out = self.res4b_1_bn(out)
        out = self.res4b_1_relu(out)

        out = self.res4b_2(out)

        out += residual2

        out = self.res4b_bn(out)
        out = self.res4b_relu(out)

        return out

class Resnet_3D_5(nn.Module):
    def __init__(self):
        super(Resnet_3D_5, self).__init__()

        self.res5a_1 = nn.Conv3d(
            256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )
        self.res5a_1_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res5a_1_relu = nn.ReLU(inplace=True)
        self.res5a_2 = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

        self.res5a_down = nn.Conv3d(
            256, 512, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)
        )

        self.res5a_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res5a_relu = nn.ReLU(inplace=True)

        self.res5b_1 = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )
        self.res5b_1_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res5b_1_relu = nn.ReLU(inplace=True)
        self.res5b_2 = nn.Conv3d(
            512, 512, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)
        )

        self.res5b_bn = nn.BatchNorm3d(
            512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True
        )
        self.res5b_relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = self.res5a_down(x)

        out = self.res5a_1(x)
        out = self.res5a_1_bn(out)
        out = self.res5a_1_relu(out)

        out = self.res5a_2(out)

        out += residual

        residual2 = out

        out = self.res5a_bn(out)
        out = self.res5a_relu(out)

        out = self.res5b_1(out)
        out = self.res5b_1_bn(out)
        out = self.res5b_1_relu(out)

        out = self.res5b_2(out)

        out += residual2

        out = self.res5b_bn(out)
        out = self.res5b_relu(out)

        return out

class ECO_3D(nn.Module):
    def __init__(self):
        super(ECO_3D, self).__init__()

        # 3D_Resnet 모듈
        self.res_3d_3 = Resnet_3D_3()
        self.res_3d_4 = Resnet_3D_4()
        self.res_3d_5 = Resnet_3D_5()

        # Global Average Pooling
        self.global_pool = nn.AvgPool3d(
            kernel_size=(4, 7, 7), stride=1, padding=0)

    def forward(self, x):
        '''
        입력 x의 크기 torch.Size([batch_num,frames, 96, 28, 28])
        '''
        out = torch.transpose(x, 1, 2)  # 텐서의 순서를 교체
        out = self.res_3d_3(out)
        out = self.res_3d_4(out)
        out = self.res_3d_5(out)
        out = self.global_pool(out)
        
        # 텐서 크기를 변경
        # torch.Size([batch_num, 512, 1, 1, 1])에서 torch.Size([batch_num, 512])로
        out =out.view(out.size()[0], out.size()[1])
        
        return out

if __name__ == '__main__':

    # Resnet_3D_3 Check
    input_1 = torch.randn(1, 96, 16, 28, 28)
    res3d3 = Resnet_3D_3()
    output_1 = res3d3(input_1)
    print(f'Resnet_3D_3 result --> {output_1.shape} == (1, 128, 16, 28, 28) ')

    # Resnet_3D_4 Check
    input_2 = torch.randn(1, 128, 16, 28, 28)
    res3d4 = Resnet_3D_4()
    output_2 = res3d4(input_2)
    print(f'Resnet_3D_4 result --> {output_2.shape} == (1, 256, 8, 14, 14) ')

    # Resnet_3D_5 Check
    input_3 = torch.randn(1, 256, 8, 14, 14)
    res3d5 = Resnet_3D_5()
    output_3 = res3d5(input_3)
    print(f'Resnet_3D_5 result --> {output_3.shape} == (1, 512, 4, 7, 7) ')

    # ECO_3D Check
    input_4 = torch.randn(1, 16, 96, 28, 28)
    eco_3d = ECO_3D()
    output_4 = eco_3d(input_4)
    print(f'ECO_3D --> {output_4.shape} == (1, 512) ')
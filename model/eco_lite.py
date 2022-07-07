import torch
import torch.nn as nn

from .eco2d import ECO_2D
from .eco3d import ECO_3D

class ECO_Lite(nn.Module):
    def __init__(self, num_of_classes=101):
        super(ECO_Lite, self).__init__()

        # 2D Net 모듈
        self.eco_2d = ECO_2D()

        # 3D Net 모듈
        self.eco_3d = ECO_3D()

        # 클래스 분류의 전결합층
        self.fc_final = nn.Linear(in_features=512, out_features=num_of_classes, bias=True)

    def forward(self, x):
        '''
        입력 x는 torch.Size([batch_num, num_segments=16, 3, 224, 224]))
        '''

        # 입력 x의 각 차원의 크기를 취득
        bs, ns, c, h, w = x.shape

        # x를 (bs*ns, c, h, w)로 크기를 변환한다
        out = x.view(-1, c, h, w)
        # (주석)
        # PyTorch의 Conv2D는 입력 크기가 (batch_num, c, h, w)만 허용되므로
        # (batch_num, num_segments, c, h, w)는 처리할 수 없다
        # 지금은 2차원 화상을 따로 처리하므로, num_segments는 batch_num의 차원에 넣어도 좋으므로
        # (batch_num×num_segments, c, h, w)로 크기를 변환한다

        # 2D Net 모듈 출력 torch.Size([batch_num×16, 96, 28, 28])
        out = self.eco_2d(out)

        # 2차원 화상을 텐서로 3차원용으로 변환
        # num_segments를 batch_num의 차원으로 넣은 것을 원래대로 되돌림
        out = out.view(-1, ns, 96, 28, 28)

        # 3D Net 모듈 출력 torch.Size([batch_num, 512])
        out = self.eco_3d(out)

        # 클래스 분류의 전결합층 출력 torch.Size([batch_num, class_num=400])
        out = self.fc_final(out)

        return out

if __name__ == '__main__':
    dummy = torch.randn(1, 16, 3, 224, 224)
    model = ECO_Lite()
    output = model(dummy)
    print(output.shape)
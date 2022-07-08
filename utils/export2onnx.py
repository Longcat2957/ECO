import os
import sys
import argparse
import torch
import onnx

from ..model.eco_lite import ECO_Lite


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, required=True, help="Number of classes")
    parser.add_argument("--weights", type=str, required=True, help="PModel Wights(Pytorch) *.pth")
    parser.add_argument("--onnx", type=str, default='eco.onnx', required=True, help="ONNX file path, default = eco.onnx")
    parser.add_argument("--log", type=bool, default=False, help="Get log Message")
    parser.add_argument("--version", type=int, default=12, help="ONNX opset_version")
    print('Pytorch 2 TensorRT export')

    opt = parser.parse_args()

    if opt.log:
        print('Export ECO model to ONNX')
        print(f'Number of classes = {opt.classes}')
        print(f'Pytorch Model Path = {opt.weights}')
        print(f'ONNX Model Path = {opt.onnx}')

    torch_model_path = opt.weights
    my_model = ECO_Lite(number_of_classes=opt.classes)

    my_model = torch.load(torch_model_path)
    my_model.eval()

    dummy = torch.randn(1, 16, 3, 224, 224)
    out = my_model(dummy)
    print(f'output.shape = {out.shape}')

    torch.onnx.export(my_model, dummy, opt.onnx, export_params=True, opset_version=opt.version, verbose=opt.log, input_names='input', output_names='output')
    print('done')



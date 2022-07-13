import torch
import torchvision
import numpy as np

import argparse

from model.eco_lite import ECO_Lite
from utils.preprocessor import preprocECO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", type=int, default=101, help='number of classes')
    parser.add_argument("--weight", type=str, help='path of weight')
    parser.add_argument("--video", type=str, help='path of video file')
    parser.add_argument("--ans", type=int, default=5, help='max answers')

    opt = parser.parse_args()

    model = ECO_Lite(opt.classes)
    vid_transformer = preprocECO()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'device = {device}')
    model.load_state_dict(torch.load(opt.weight))
    model.eval()
    model.to(device)
    
    my_input = vid_transformer(opt.video)
    my_input.to(device)

    with torch.set_grad_enabled(False):
        output = model(my_input)
    
    output.to('cpu')
    output_clone = output.clone()

    for i in range(opt.ans):
        _, pred = torch.max(output_clone, dim=0)
        class_idx = int(pred.numpy())
        print(f"예측 {i+1}위 \t {class_idx}")
        output_clone[class_idx] = -1000

    print('done')
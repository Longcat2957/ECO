import os
import cv2
import argparse
import pathlib

def make_datapath_list(root_path='../data/UCF-101/'):
    """
    return folder_list
    root_path : str, root_path of data ('~/data/UCF-101')
    """
    video_list = []
    class_list = os.listdir(path=root_path)

    for class_list_i in (class_list):
        class_path = os.path.join(root_path, class_list_i)

        for file_name in os.listdir(class_path):
            name, ext = os.path.splitext(file_name) # ignore formats(*.avi)
            video_img_dir = os.path.join(class_path, name)

            video_list.append(video_img_dir)

    return video_list

if __name__ == '__main__':
    """
    특정 갯수 이하의 이미지를 가진 폴더를 찾아낸다. (직접 지우지는 않음)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default="../../data/UCF-101/train/")
    parser.add_argument("--length", type=int, default=30)
    opt = parser.parse_args()
    # *.avi foramt to *.jpg

    root_path = opt.dir
    parent_path = os.listdir(root_path)
    for p in parent_path:
        parent_dir = os.path.join(root_path, p)
        baby_path = os.listdir(parent_dir)
        
        for b in baby_path:
            baby_dir = os.path.join(parent_dir, b)
            file_number = len(os.listdir(baby_dir))

            if file_number <= opt.length:
                print(baby_dir)
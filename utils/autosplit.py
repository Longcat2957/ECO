import os
import argparse
import pathlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="../../data/UCF-101")

    opt = parser.parse_args()
    root_path = pathlib.Path(opt.root_dir)

    class_list = os.listdir(path=root_path)
    class_number = len(class_list)
    print(f'Total Class number = {class_number}')

    train_dir = os.path.join(root_path, 'train')
    val_dir = os.path.join(root_path, 'val')

    if not os.path.exists(train_dir):
        print(f'make directory {train_dir}')
        os.mkdir(train_dir)
    
    if not os.path.exists(val_dir):
        print(f'make directory {val_dir}')
        os.mkdir(val_dir)
    
    # in progress...
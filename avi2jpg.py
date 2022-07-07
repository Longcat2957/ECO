import os
import subprocess
import cv2

class avi2jpg:
    vid = None
    
    def convert(self, video, save_path):
        vidcap = cv2.VideoCapture(video)
        success,image = vidcap.read()
        count = 0
        while success:
            framecount = "image_{number:05}".format(number=count)
            save_dir = os.path.join(save_path, framecount)
            cv2.imwrite(save_dir+".jpg", image)     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1


if __name__ == '__main__':
    # *.avi foramt to *.jpg
    dir_path = '../data/UCF-101/val'
    class_list = os.listdir(path=dir_path)
    # print(class_list)
    
    # avi2jpg
    converter = avi2jpg()

    for class_list_i in (class_list):
        class_path = os.path.join(dir_path, class_list_i)

        for file_name in os.listdir(class_path):

            # filename, format
            name, ext = os.path.splitext(file_name)

            dst_directory_path = os.path.join(class_path, name)
            print(dst_directory_path)
            if not os.path.exists(dst_directory_path):
                os.mkdir(dst_directory_path)
            
            # path of video file
            video_file_path = os.path.join(class_path, file_name)
            converter.convert(video_file_path, dst_directory_path)

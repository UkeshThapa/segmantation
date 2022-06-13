import cv2
import os
import glob
import numpy as np


pattern = r'E:\Data\X-ray_Dataset\masks' + "\*"
list_name_images =[]
for file in glob.iglob(pattern, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    img = cv2.imread(f"E:/Data/X-ray_Dataset/masks/{file_name}")
    img_np = np.asarray(img)
    # gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite(f'E:/Data/X-ray_Dataset/new_masks_channel_1/{file_name}', gray_image)
    # img_np = np.asarray(gray_img)
    print(img_np.shape)


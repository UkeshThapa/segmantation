import cv2
import glob
import os
import numpy as np






pattern = r'E:\Data\processed_image\After_flaged\montgomery_xray'+'\*' 

for file in glob.iglob(pattern, recursive=True):    

    file_name = os.path.basename(file)
    # name,dot,ext=file_name.partition('.')
    image = cv2.imread(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((8,8), np.uint8)
    dilated = cv2.dilate(gray, kernel,  iterations=5)
    cv2.imwrite(f"E:/Data/processed_image/after_morph/montgomery_xray/{file_name}",dilated)

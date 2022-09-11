# importing libraries
import cv2
import glob
import os
import shutil
import numpy as np

# reading the image data from desired directory\
src_folder = r"E:\Data\processed_image\post-process_output_xray\NIH_large_dataset"
pattern = src_folder + "\*"
list_per=[]

for file in glob.iglob(pattern, recursive=True):
    img=cv2.imread(file)
    # extract file name form file path
# counting the number of pixels
    number_of_white_pix = np.sum(img == 255)
    number_of_black_pix = np.sum(img == 0)
    percent = (number_of_white_pix /(number_of_white_pix+number_of_black_pix))*100
    print(percent)
    list_per.append(percent)

mean = np.mean(np.array(list_per))

stad = np.std(np.array(list_per))
print("mean of NIH",mean)
print("standard deviation",stad)
# img=cv2.imread('1.png')
# list_per =[24,40,35]
# number_of_white_pix = np.sum(img == 255)
# number_of_black_pix = np.sum(img == 0)
# percent = (number_of_white_pix /(number_of_white_pix+number_of_black_pix))*100
# list_per.append(percent)
# lists_per = np.array(list_per)
# print(np.mean(lists_per))
# print(np.std(lists_per))

# src_folder = r"D:\Maya.Ai Reseacrh Center\x-ray\Dataset\covid-19-chest-x-ray-dataset\new_masks"
# dst_folder = r"D:\Maya.Ai Reseacrh Center\x-ray\Dataset\covid-19-chest-x-ray-dataset\small_object_masks\\"

# pattern = src_folder + "\*"

# for file in glob.iglob(pattern, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     if file_name in list_name:
#         shutil.move(file, dst_folder + file_name)
#         print('Moved:', file)

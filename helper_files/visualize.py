import os
import glob
import cv2
import numpy as np
import shutil

import random



# random.seed(100)

# pattern = r'E:\Data\NIH_dataset\images' + "\*"
# list_name_images =[]
# for file in glob.iglob(pattern, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     # name , extension = os.path.splitext(file_name)
#     list_name_images.append(file_name)

# print(len(list_name_images))

# random_train_val = random.sample(list_name_images, int(len(list_name_images)*0.008))


# pattern = r'E:\Data\NIH_dataset\images' + "\*"
# pred =r'E:\Results\Segmentation\NIH_dataset\\'
# list_name_masks=[]
# for file in glob.iglob(pattern, recursive=True):
#     # extract file name form file path

#     file_name = os.path.basename(file)
#     if file_name in random_train_val:
        
#         background = cv2.imread(file)
#         # name,dot,ext=file_name.partition('_mask')
#         overlay = cv2.imread(pred+file_name)

#         overlay[np.all(overlay==(255,255,255), axis=-1)] = (0,255,0)
#         # pred_img
#         added_image = cv2.addWeighted(background,0.8,overlay,0.08,1)

        # img_pred
        # added_image = cv2.addWeighted(background,0.9123,overlay,0.099,1)

        # GT_img
        # added_image = cv2.addWeighted(background,0.9123,overlay,0.099,1)
        # cv2.imwrite(f'E:/Data/processed_image/visualize_dataset/NIH_large_dataset/{file_name}', added_image)

# pattern = r'E:\Data\processed_image\Orignal_4_Dataset\NIH\masks' + "\*"
# pred =r'E:\Data\processed_image\post-process_output_xray\NIH-xray\\'
# list_name_masks=[]
# for file in glob.iglob(pattern, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     background = cv2.imread(file)
    
#     name,dot,ext=file_name.partition('_mask')

#     overlay = cv2.imread(pred+name+'.png')

#     overlay[np.all(overlay==(255,255,255), axis=-1)] = (0,0,255)
#     # pred_img
#     # added_image = cv2.addWeighted(background,0.5,overlay,0.3,1)

#     # img_pred
#     # added_image = cv2.addWeighted(background,0.9123,overlay,0.099,1)

#     # GT_img
#     added_image = cv2.addWeighted(background,0.5,overlay,0.3,1)
#     cv2.imwrite(f'E:/Data/processed_image/visualize_dataset/NIH_visualize/Gt_PRED/{name}.png', added_image)



# pattern = r'E:\Data\processed_image\Orignal_4_Dataset\NIH\\' 
# pred =r'E:\Data\processed_image\post-process_output_xray\NIH-xrays'+ "\*"
# list_name_masks=[]
# for file in glob.iglob(pred, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     print('aayo')
#     name,dot,ext=file_name.partition('.')
#     overlay = cv2.imread(file)
#     background = cv2.imread(pattern+name+'_mask.png')
#     overlay[np.all(overlay==(255,255,255), axis=-1)] = (0,0,255)
#     print(overlay)
#     # pred_img
#     # added_image = cv2.addWeighted(background,0.5,overlay,0.3,1)

#     # Gt_pred
#     # added_image = cv2.addWeighted(background,0.9123,overlay,0.099,1)
#     added_image = cv2.addWeighted(background,0.5,overlay,0.3,1)
#     cv2.imshow('combine',added_image)
#     # GT_img
#     # added_image = cv2.addWeighted(background,0.9123,overlay,0.099,1)
#     cv2.imwrite(f'E:/Data/processed_image/visualize_dataset/NIH_visualize/Gt_pred/{file_name}', added_image)

# background = cv2.imread('NIH_0001_mask.png')
# overlay = cv2.imread('NIH_0001_pred.png')

# overlay[np.all(overlay==(255,255,255), axis=-1)] = (0,0,255)

# added_image = cv2.addWeighted(background,0.9123,overlay,0.099,1)

# cv2.imwrite('combine2.png',added_image)

# import pydicom
# from pydicom.pixel_data_handlers.util import apply_voi_lut
# def read_xray(path, voi_lut = True, fix_monochrome = True):
#     # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
#     dicom = pydicom.read_file(path)
#     # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
#     # "human-friendly" view
#     if voi_lut:
#         data = apply_voi_lut(dicom.pixel_array, dicom)
#     else:
#         data = dicom.pixel_array
#     # depending on this value, X-ray may look inverted - fix that:
#     if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
#         data = np.amax(data) - data
#     data = data - np.min(data)
#     data = data / np.max(data)
#     data = (data * 255).astype(np.uint8)
#     return data


path=r'E:\Data\Dataset\Processed_image\vinbigdata\images'+'\*'



for file in glob.iglob(path, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    name,dot,other=file_name.partition('.dicom')
    print(name)
    os.rename(file,f'E:/Data/Dataset/Processed_image/vinbigdata/images/{name}.png')
    # shutil.copy(file,f'E:/Data/Dataset/Processed_image/vinbigdata/new_images/{name}.png')
    # try:
        # img=read_xray(file)
        # cv2.imwrite(f'E:/Data/Dataset/Processed_image/vinbigdata/images/{file_name}.png',img)
    # except:
    #     txt = open('dicom_name.txt','a')
    #     txt.write(str(file_name)+'\n')
    #     txt.close()
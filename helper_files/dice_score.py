import os
from PIL import Image
import numpy as np
import pandas as pd
import cv2 as cv
import glob
import csv

def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 



""" for all datasets dice score"""

# dataset_names=['nih_xrays_dataset','montgomery_xrays_dataset','japan_xrays_dataset','china_xrays_dataset','darwinlungs','u4_dataset','u5_dataset']
# df = pd.read_csv('dice_score.csv')
 
# for i in range(len(dataset_names)):
#   nam = dataset_names[i]
#   list_name=[]
#   list_name.append(dataset_names[i])
#   print(f'Checking with {nam}')
#   txt = open('dice_score1.txt','a')
#   txt.write('\n'+f'Checking with {nam}'+'\n')
#   txt.close()
#   for sub_name in dataset_names:

#       prd_path = f'E:\\Data\\Dataset\\Results\\Segmentation\\{nam}_model\\{sub_name}\\'
#       gt_path = f'E:\\Data\\Dataset\\Processed_image\\lungs-segmentation-dataset\\{sub_name}\\masks\\'
#       prd_list = os.listdir(prd_path)
#       mean_dice = []

#       for fname in prd_list:
#           name,dot,ext=fname.partition('.')
#           try:
#               gt = np.array(Image.open(gt_path+name+'_mask.png'))
#               pred = np.array(Image.open(prd_path+fname))
#               # pred[pred==255]=1
#               print("dice score is :",fname,compute_dice_coefficient(gt,pred))
#               mean_dice.append(compute_dice_coefficient(gt,pred))

#           except:
#               gt = np.array(Image.open(gt_path+name+'.png'))
#               pred = np.array(Image.open(prd_path+fname))
#               # pred[pred==255]=1
#               print("dice score is :",fname,compute_dice_coefficient(gt,pred))
#               mean_dice.append(compute_dice_coefficient(gt,pred))

#       m_dice = sum(mean_dice)/len(mean_dice)
#       list_name.append(m_dice)
#       print(f'---> average dice score{sub_name}',m_dice)
#       txt = open('dice_score1.txt','a')
#       txt.write(f'---> average dice score {sub_name}  :  {m_dice}'+'\n')
#       txt.close()
#   data=[list_name[0],list_name[1],list_name[2],list_name[3],list_name[4],list_name[5],list_name[6],list_name[7]]    
#   with open('dice_score1.csv', 'a', encoding='UTF8',newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(data)

''''china_xray'''


'''Dice score of orignal images'''
prd_path = r'E:\Data\Dataset\Results\Segmentation\montgomery_xrays_dataset\\'
gt_path = r'E:\Data\Dataset\Processed_image\lungs-segmentation-dataset\montgomery_xrays_dataset\masks\\'
prd_list = os.listdir(prd_path)
mean_dice = []

for fname in prd_list:
    name,dot,ext=fname.partition('.')
    gt = np.array(Image.open(gt_path+name+'.png'))
    pred = np.array(Image.open(prd_path+fname))
    # pred[pred==255]=1
    print("dice score is :",fname,compute_dice_coefficient(gt,pred))
    mean_dice.append(compute_dice_coefficient(gt,pred))

c_org = sum(mean_dice)/len(mean_dice)

print('average dice score',c_org)





# path = r'E:\Data\Dataset\Processed_image\lungs-segmentation-dataset\montgomery_xrays_dataset\images'+'\*'

# for file in glob.iglob(path, recursive=True):    
#     img = cv.imread(file)
#     h = int(img.shape[0]*0.80)
#     w = img.shape[1]
#     channels = img.shape[2]
#     file_name = os.path.basename(file)
#     crop_img = img[0 : h, 0 : w]
    
#     cv.imwrite(f'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/montgomery_xrays_dataset/crop_image/{file_name}', crop_img)
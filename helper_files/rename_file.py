import random
import glob
import os
import cv2
from cv2 import imwrite


pattern = r'E:\Data\Output_xray\NIH_xrays_dataset' + "\*"
list_name_masks=[]
for file in glob.iglob(pattern, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    name,other,ext=file_name.partition('.')
    name,dot,exts=ext.partition('.')
    print(name)
    img = cv2.imread(file)
    gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"E:/Data/Output_xray/NIH_xray/{file_name}",gray, [cv2.IMWRITE_PNG_BILEVEL, 1])



# patterns = r'E:\Data\processed_image\Orignal_4_Dataset\Chain_x-ray\images' + "\*"
# for file in glob.iglob(patterns, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     name,dash,other=file_name.partition('.')
#     img=cv2.imread(file)
#     if name in list_name_masks:
#         print(name)
#         cv2.imwrite(f'E:/Data/processed_image/Orignal_4_Dataset/Chain_x-ray/images_masks/{file_name}',img)
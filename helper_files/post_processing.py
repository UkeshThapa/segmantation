"""
Segmentation
find two large component
    1. single component
    2. two component check if component is less than threshold (flag the images)
take the co-ordinate in csv file
use csv file to crop the images
    1. if the images is less than threshold take the whole images
    2. images must all frontal

"""
import glob
import os
import cv2 
import numpy as np
import shutil
import pandas as pd


# df = pd.read_csv('NIH.csv')
# print(df.head(5))
# label = df['Finding Labels'].value_counts()
# print(df['Finding Labels'].value_counts().sum())
# print(label)
# df['Finding Labels'].value_counts().reset_index().to_csv('df.csv')

'''
    find the two large component
'''
# def two_large_component(src,dst):
#     for file in glob.iglob(src, recursive=True):    
#         file_name = os.path.basename(file)
#         try:    
#             img = cv2.imread(file)
#             gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


#             nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)

#             sizes = stats[:, -1]
#             new_dict ={}
#             for i in range(1,nb_components):
#                 new_dict[f'{i}'] = sizes[i]
#             max_value = sorted(new_dict.values())[-1]
#             Sec_max_value = sorted(new_dict.values())[-2]

#             for keys, val in new_dict.items():
#                 if val == max_value:
#                     img2 = np.zeros(output.shape)
#                     img2[output==int(keys)] = 255

#             for keys, val in new_dict.items():
#                 if val == Sec_max_value:
#                     img3 = np.zeros(output.shape)
#                     img3[output==int(keys)] = 255
#             img4=img2+img3
#             print(file_name)
#             cv2.imwrite(f"{dst}/{file_name}", img4, [cv2.IMWRITE_PNG_BILEVEL, 1])
#         except:
#             test=open('image_name.txt','w')
#             test.write(file_name+'\n')
#             test.close()


# src=r''
# dst=''
# two_large_component(src,dst)


'''FIND THE MEAN AND STD OF THE MASK IMAGES'''
# src = r"E:\Data\processed_image\post-process_output_xray\NIH_large_dataset"+ "\*"
# # arr = np.array([])

# for file in glob.iglob( src, recursive=True):

#     img=cv2.imread(file,0)
#     file_name = os.path.basename(file)
#     height ,width= img.shape
#     percentage=(cv2.countNonZero(img)/(height*width))*100
# #     # arr = np.append(arr, percentage)
# #     # when std dev is 2
# #     if percentage < 13.5:
# #         txt = open('2std.txt','a')
# #         txt.write(str(file_name)+str(percentage)+'\n')
# #         txt.close()
#     if percentage < 7:
#         txt = open('3std1.txt','a')
#         txt.write(str(file_name)+str(percentage)+'\n')
#         txt.close()


# print(np.mean(arr))
# print(np.std(arr))



" save the coordinate"


# import csv  
# import time

# start_time = time.time()
# header = ['Name', 'start_x_cordinate', 'start_y_cordinate', 'end_x_cordinate','end_y_cordinate']


# with open('cheXpert_cordinate.csv', 'a', encoding='UTF8',newline='') as f:
#     writer = csv.writer(f)

#     # write the header
#     writer.writerow(header)

    # write the data
    # writer.writerow(data)


# print("--- %s seconds ---" % (((time.time() - start_time)*112078)/60))




# ''' data split for train and save in another folder '''
with open(os.path.join(os.path.join('E:\projects\\X-ray\\Deeplab-Xception-Lungs-Segmentation-master\\helper_files\\test_actual.txt')), "r") as f:
    lines = f.read().splitlines()

list_name = []
for ii, line in enumerate(lines):

    name ,dot,ext = line.partition(' ')
    list_name.append(name)

list_train = []
paths =r'E:\Data\Dataset\Processed_image\Crop_images\NIH_dataset'+'\*'
# dst_folders = r"E:\Data\NIH_dataset\images_train\\"
for file in glob.iglob(paths, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    if file_name in list_name:
        list_train.append(file_name)

for ii, line in enumerate(lines):

    name ,dot,ext = line.partition(' ')
    if name in list_train:
        txt = open('test_nih_crop.txt','a')
        txt.write(str(line)+'\n')
        txt.close()



# paths =r'E:\Data\NIH_dataset\images'+'\*'
# dst_folders = r"E:\Data\NIH_dataset\not_crop\\"
# for file in glob.iglob(paths, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     if file_name not in list_name:
#         txt = open('notcrop.txt','a')
#         txt.write(str(file_name)+'\n')
#         txt.close()

#         print(file_name)
#         shutil.copy(file, dst_folders + file_name)

# paths =r'E:\Data\processed_image\post-process_output_xray\NIH_large_dataset'+'\*'
# dst_folders = r"E:\Data\NIH_dataset\not_crop_mask\\"
# for file in glob.iglob(paths, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     if file_name not in list_name:
#         print(file_name)
#         shutil.copy(file, dst_folders + file_name)



# img=cv2.imread('3.png',0)

# height ,width= img.shape
# percentage=(cv2.countNonZero(img)/(height*width))*100
# print(percentage)






'''
Connect component with opencv 

cv2.connectedComponentsWithStats returns 
    1. bounding box of the component
    2. area of the components
    3. centroid coordinate of the components


'''
# importing the necessary package
import glob
import os
import cv2 
import numpy as np

'''

find the two largest connected components

'''
# pattern = r'E:\Results\Segmentation\NIH_dataset' + "\*"

# for file in glob.iglob(pattern, recursive=True):    
#     file_name = os.path.basename(file)

#     img = cv2.imread(file)
#     gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#     no_pred=[]
#     try:
#         nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)

#         sizes = stats[:, -1]
#         new_dict ={}
#         for i in range(1,nb_components):
#             new_dict[f'{i}'] = sizes[i]
#         max_value = sorted(new_dict.values())[-1]
#         Sec_max_value = sorted(new_dict.values())[-2]

#         for keys, val in new_dict.items():
#             if val == max_value:
#                 img2 = np.zeros(output.shape)
#                 img2[output==int(keys)] = 255

#         for keys, val in new_dict.items():
#             if val == Sec_max_value:
#                 img3 = np.zeros(output.shape)
#                 img3[output==int(keys)] = 255
#         img4=img2+img3
#         print(file_name)
#         cv2.imwrite(f"E:/Data/processed_image/post-process_output_xray/NIH_large_dataset/{file_name}", img4, [cv2.IMWRITE_PNG_BILEVEL, 1])
#     except:
#         no_pred.append(file_name)
#         print('not prd',file_name)
# text_file1 = open("issue.txt", "w")

# for i in range(len(no_pred)):

 
#     text_file1.write(no_pred[i])
#     text_file1.write('\n')
# #close file
# text_file1.close()
'''
flag the connected component which has size less than 50% 

'''


# pattern = r'E:\Data\processed_image\post-process_output_xray\Montgomery_xrays' + "\*"
# pred =r'E:\Data\Output_xray\Montgomery_xrays_dataset\\'
# for file in glob.iglob(pattern, recursive=True):    
#     file_name = os.path.basename(file)

#     img = cv2.imread(file)
#     pred_mask = cv2.imread(pred+file_name)
#     pred_gray = cv2.cvtColor(pred_mask , cv2.COLOR_BGR2GRAY)

#     gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#     # print(gray)

#     nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)

#     sizes = stats[:, -1]
#     new_dict ={}
#     for i in range(1,nb_components):
#         new_dict[f'{i}'] = sizes[i]
#     # print(new_dict)
#     max_value = sorted(new_dict.values())[-1]
#     Sec_max_value = sorted(new_dict.values())[-2]
#     # print(max_value,Sec_max_value)
#     diff = max_value- Sec_max_value
#     # print((diff/max_value)*100)
#     if (diff/max_value)*100 > 25:
#             print(file_name)
#             print((diff/max_value)*100)
#             cv2.imwrite(f"E:/Data/processed_image/flaged/montgomery-xray/{file_name}",pred_gray, [cv2.IMWRITE_PNG_BILEVEL, 1])
#     else:
#         print('done')

    # for keys, val in new_dict.items():
    #   if val == max_value:
    #     img2 = np.zeros(output.shape)
    #     img2[output==int(keys)] = 255

    # for keys, val in new_dict.items():
    #   if val == Sec_max_value:
    #     img3 = np.zeros(output.shape)
    #     img3[output==int(keys)] = 255
    # img4=img2+img3
    # print(file_name)
    # cv2.imwrite(f"E:/Data/processed_image/post-process_output_xray/darwin_lungs/{file_name}", img4, [cv2.IMWRITE_PNG_BILEVEL, 1])



# pattern = r'E:\Data\processed_image\post-process_output_xray\Montgomery_xrays' + "\*"
# pred =r'E:\Data\Output_xray\Montgomery_xrays_dataset\\'
# for file in glob.iglob(pattern, recursive=True):    
#     file_name = os.path.basename(file)

#     img = cv2.imread(file)
#     pred_mask = cv2.imread(pred+file_name)
#     pred_gray = cv2.cvtColor(pred_mask , cv2.COLOR_BGR2GRAY)

#     gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#     # print(gray)

#     nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)

#     sizes = stats[:, -1]
#     new_dict ={}
#     for i in range(1,nb_components):
#         new_dict[f'{i}'] = sizes[i]
#     # print(new_dict)
#     max_value = sorted(new_dict.values())[-1]
#     Sec_max_value = sorted(new_dict.values())[-2]
#     # print(max_value,Sec_max_value)
#     diff = max_value- Sec_max_value
#     # print((diff/max_value)*100)
#     if (diff/max_value)*100 > 25:
#             print(file_name)
#             print((diff/max_value)*100)
#             cv2.imwrite(f"E:/Data/processed_image/flaged/montgomery-xray/{file_name}",pred_gray, [cv2.IMWRITE_PNG_BILEVEL, 1])
#     else:
#         print('done')

    # for keys, val in new_dict.items():
    #   if val == max_value:
    #     img2 = np.zeros(output.shape)
    #     img2[output==int(keys)] = 255

    # for keys, val in new_dict.items():
    #   if val == Sec_max_value:
    #     img3 = np.zeros(output.shape)
    #     img3[output==int(keys)] = 255
    # img4=img2+img3
    # print(file_name)
    # cv2.imwrite(f"E:/Data/processed_image/post-process_output_xray/darwin_lungs/{file_name}", img4, [cv2.IMWRITE_PNG_BILEVEL, 1])

#








''' Connected component boundary with opencv '''




# post_mask = r'E:\Data\processed_image\post-process_output_xray\NIH-xray\\' 
# pred_mask =r'E:\Data\processed_image\flaged\NIH-xray'+ "\*"

# for file in glob.iglob(pred_mask, recursive=True):    
#     file_name = os.path.basename(file)

#     # img = cv2.imread(file)
#     # pred_mask = cv2.imread(file_name)

#     post_mask_img = cv2.imread(post_mask+file_name)
#     pred_mask_img = cv2.imread(file)
#     gray = cv2.cvtColor(post_mask_img , cv2.COLOR_BGR2GRAY)
#     pred_gray = cv2.cvtColor(pred_mask_img , cv2.COLOR_BGR2GRAY)
#     new_dic={}

#     output= cv2.connectedComponentsWithStats(gray)
#     nb_components, output, stats,centroid = output

#     for i in range(1,nb_components):
#         x = stats[i, cv2.CC_STAT_LEFT]
#         y = stats[i, cv2.CC_STAT_TOP]
#         w = stats[i, cv2.CC_STAT_WIDTH]
#         h = stats[i, cv2.CC_STAT_HEIGHT]
#         area = stats[i, cv2.CC_STAT_AREA]
#         (cX, cY) = centroid[i]
#         # print(x,y)
#         # print(w,h)
#         # print(area)
#         new_dic[f'x{i}'] = x
#         new_dic[f'y{i}'] = y
#         new_dic[f'w{i}'] = w
#         new_dic[f'h{i}'] = h
#         new_dic[f'A{i}'] = area
#         new_dic[f'cx{i}'] = cX
#         new_dic[f'cy{i}'] = cY


#     # print(new_dic)


#     # left or right lungs
#     left_lungs={}
#     right_lungs={}

#     if new_dic['x1']<new_dic['x2']:
#         left_lungs[f'x'] = new_dic['x1']
#         left_lungs[f'y'] = new_dic['y1']
#         left_lungs[f'w'] = new_dic['w1']
#         left_lungs[f'h'] = new_dic['h1']
#         left_lungs[f'A'] = new_dic['A1']
#         left_lungs[f'cx'] = int(new_dic['cx1']//1)
#     else:
#         left_lungs[f'x'] = new_dic['x2']
#         left_lungs[f'y'] = new_dic['y2']
#         left_lungs[f'w'] = new_dic['w2']
#         left_lungs[f'h'] = new_dic['h2']
#         left_lungs[f'A'] = new_dic['A2']
#         left_lungs[f'cx'] = int(new_dic['cx2']//1)

#     if new_dic['x1'] > new_dic['x2']:
#         right_lungs[f'x'] = new_dic['x1']
#         right_lungs[f'y'] = new_dic['y1']
#         right_lungs[f'w'] = new_dic['w1']
#         right_lungs[f'h'] = new_dic['h1']
#         right_lungs[f'A'] = new_dic['A1']
#         right_lungs[f'cx'] = int(new_dic['cx1']//1)

#     else:
#         right_lungs[f'x'] = new_dic['x2']
#         right_lungs[f'y'] = new_dic['y2']
#         right_lungs[f'w'] = new_dic['w2']
#         right_lungs[f'h'] = new_dic['h2']
#         right_lungs[f'A'] = new_dic['A2']
#         right_lungs[f'cx'] = int(new_dic['cx2']//1)




#     y1 =left_lungs['y'] if left_lungs['y'] < right_lungs['y'] else right_lungs['y']
#     w =left_lungs['w'] if left_lungs['w'] > right_lungs['w'] else right_lungs['w']
#     h =left_lungs['h'] if left_lungs['h'] > right_lungs['h'] else right_lungs['h']
#     print(y,w,h)


#     if left_lungs['A']>right_lungs['A']:

#         mask = np.zeros_like(pred_gray)
#         mask=cv2.rectangle(mask, (left_lungs['x'], y1), (right_lungs['cx']+(w//2), y1+ h), (255, 255, 255), -1)
#         result = cv2.bitwise_and(pred_gray, mask)
#         cv2.imwrite(f'E:/Data/processed_image/After_flaged/NIH_xray/{file_name}',result, [cv2.IMWRITE_PNG_BILEVEL, 1])
#     else:
#         mask = np.zeros_like(pred_gray)
#         mask=cv2.rectangle(mask, (left_lungs['cx']-(w//2), y1), (right_lungs['x']+w, y1+ h), (255, 255, 255), -1)
#         result = cv2.bitwise_and(pred_gray, mask)
#         cv2.imwrite(f'E:/Data/processed_image/After_flaged/NIH_xray/{file_name}',result, [cv2.IMWRITE_PNG_BILEVEL, 1])





''''crop the connected component image'''




# pattern = r'E:\Data\NIH_dataset\crop_images' + "\*"
# # dst_folder = r"E:\Data\NIH_dataset\flag_seg\\"
# list_name_images =[]
# for file in glob.iglob(pattern, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     # name , extension = os.path.splitext(file_name)
    
#     list_name_images.append(file_name)

# pred_mask = r'E:\Data\NIH_dataset\not_crop_mask\\' 
# img_path =r'E:\Data\NIH_dataset\not_crop'+ "\*"
# list_name = []
# for file in glob.iglob(img_path, recursive=True):    
#     file_name = os.path.basename(file)
#     name,dot,ext=file_name.partition('.')
#     print(file_name)
#     img = cv2.imread(file)
#     height,width,channels = img.shape
#     mask_img = cv2.imread(pred_mask+name+'.png')
#     gray = cv2.cvtColor(mask_img , cv2.COLOR_BGR2GRAY)
#     # pred_gray = cv2.cvtColor(pred_mask_img , cv2.COLOR_BGR2GRAY)
#     new_dic={}

#     output= cv2.connectedComponentsWithStats(gray)
#     nb_components, output, stats,centroid = output

#     for i in range(1,nb_components):
#         x = stats[i, cv2.CC_STAT_LEFT]
#         y = stats[i, cv2.CC_STAT_TOP]
#         w = stats[i, cv2.CC_STAT_WIDTH]
#         h = stats[i, cv2.CC_STAT_HEIGHT]
#         area = stats[i, cv2.CC_STAT_AREA]
#         (cX, cY) = centroid[i]
#     # print(x,y)
#     # print(w,h)
#     # print(area)
#         new_dic[f'x{i}'] = x
#         new_dic[f'y{i}'] = y
#         new_dic[f'w{i}'] = w
#         new_dic[f'h{i}'] = h
#         new_dic[f'A{i}'] = area
#         new_dic[f'cx{i}'] = cX
#         new_dic[f'cy{i}'] = cY


# # print(new_dic)


# # left or right lungs
#     left_lungs={}
#     right_lungs={}

#     if new_dic['x1']<new_dic['x2']:
#         left_lungs[f'x'] = new_dic['x1']
#         left_lungs[f'y'] = new_dic['y1']
#         left_lungs[f'w'] = new_dic['w1']
#         left_lungs[f'h'] = new_dic['h1']
#         left_lungs[f'A'] = new_dic['A1']
#         left_lungs[f'cx'] = int(new_dic['cx1']//1)
#     else:
#         left_lungs[f'x'] = new_dic['x2']
#         left_lungs[f'y'] = new_dic['y2']
#         left_lungs[f'w'] = new_dic['w2']
#         left_lungs[f'h'] = new_dic['h2']
#         left_lungs[f'A'] = new_dic['A2']
#         left_lungs[f'cx'] = int(new_dic['cx2']//1)

#     if new_dic['x1'] > new_dic['x2']:
#         right_lungs[f'x'] = new_dic['x1']
#         right_lungs[f'y'] = new_dic['y1']
#         right_lungs[f'w'] = new_dic['w1']
#         right_lungs[f'h'] = new_dic['h1']
#         right_lungs[f'A'] = new_dic['A1']
#         right_lungs[f'cx'] = int(new_dic['cx1']//1)

#     else:
#         right_lungs[f'x'] = new_dic['x2']
#         right_lungs[f'y'] = new_dic['y2']
#         right_lungs[f'w'] = new_dic['w2']
#         right_lungs[f'h'] = new_dic['h2']
#         right_lungs[f'A'] = new_dic['A2']
#         right_lungs[f'cx'] = int(new_dic['cx2']//1)




#     y1 =left_lungs['y'] if left_lungs['y'] < right_lungs['y'] else right_lungs['y']
#     w =left_lungs['w'] if left_lungs['w'] > right_lungs['w'] else right_lungs['w']
#     h =left_lungs['h'] if left_lungs['h'] > right_lungs['h'] else right_lungs['h']
#     # print(y,w,h)


#     if left_lungs['A']>right_lungs['A']:
#         print('left')
#         if right_lungs['cx']+(w//2) > width:
#             crop_img = img[y1:y1+h, left_lungs['x']:width]
#         else:
#             crop_img = img[y1:y1+h, left_lungs['x']:right_lungs['cx']+(w//2)]

#         # print(name)
#         # mask = np.zeros_like(img)
#         # mask=cv2.rectangle(mask, (left_lungs['x'], y1), (right_lungs['cx']+(w//2), y1+ h), (255, 255, 255), -1)
#         # result = cv2.bitwise_and(img, mask)
#         cv2.imwrite(f'E:/Data/NIH_dataset/after_crop/{file_name}',crop_img)
#     else:
#         print('right')
#         if left_lungs['cx']-(w//2) < 0:
#             crop_img = img[y1:y1+h, 0:right_lungs['x']+w]
#         else:
#             crop_img = img[y1:y1+h, left_lungs['cx']-(w//2):right_lungs['x']+w]

#         # mask = np.zeros_like(img)
#         # mask=cv2.rectangle(mask, (left_lungs['cx']-(w//2), y1), (right_lungs['x']+w, y1+ h), (255, 255, 255), -1)
#         # result = cv2.bitwise_and(img, mask)
#         cv2.imwrite(f'E:/Data/NIH_dataset/after_crop/{file_name}',crop_img)


#     text_file1 = open("component_crop.txt", "w")

# for i in range(len(list_name)):
#     text_file1.write(list_name[i])
#     text_file1.write('\n')
#     print(i)
# #close file
# text_file1.close()



import os 
import pandas as pd


# df = pd.read_csv('co-ordinate.csv')
# print(df.head(10)) 00000945_000.png


img = cv2.imread('1.png')
mask_img = cv2.imread('2.png')
height,width,channels = img.shape

gray = cv2.cvtColor(mask_img , cv2.COLOR_BGR2GRAY)
# pred_gray = cv2.cvtColor(pred_mask_img , cv2.COLOR_BGR2GRAY)
new_dic={}

output= cv2.connectedComponentsWithStats(gray)
nb_components, output, stats,centroid = output

for i in range(1,nb_components):
    x = stats[i, cv2.CC_STAT_LEFT]
    y = stats[i, cv2.CC_STAT_TOP]
    w = stats[i, cv2.CC_STAT_WIDTH]
    h = stats[i, cv2.CC_STAT_HEIGHT]
    area = stats[i, cv2.CC_STAT_AREA]
    (cX, cY) = centroid[i]
    # print(x,y)
    # print(w,h)
    # print(area)
    new_dic[f'x{i}'] = x
    new_dic[f'y{i}'] = y
    new_dic[f'w{i}'] = w
    new_dic[f'h{i}'] = h
    new_dic[f'A{i}'] = area
    new_dic[f'cx{i}'] = cX
    new_dic[f'cy{i}'] = cY


# print(new_dic)


# left or right lungs
left_lungs={}
right_lungs={}

if new_dic['x1']<new_dic['x2']:
    left_lungs[f'x'] = new_dic['x1']
    left_lungs[f'y'] = new_dic['y1']
    left_lungs[f'w'] = new_dic['w1']
    left_lungs[f'h'] = new_dic['h1']
    left_lungs[f'A'] = new_dic['A1']
    left_lungs[f'cx'] = int(new_dic['cx1']//1)
else:
    left_lungs[f'x'] = new_dic['x2']
    left_lungs[f'y'] = new_dic['y2']
    left_lungs[f'w'] = new_dic['w2']
    left_lungs[f'h'] = new_dic['h2']
    left_lungs[f'A'] = new_dic['A2']
    left_lungs[f'cx'] = int(new_dic['cx2']//1)

if new_dic['x1'] > new_dic['x2']:
    right_lungs[f'x'] = new_dic['x1']
    right_lungs[f'y'] = new_dic['y1']
    right_lungs[f'w'] = new_dic['w1']
    right_lungs[f'h'] = new_dic['h1']
    right_lungs[f'A'] = new_dic['A1']
    right_lungs[f'cx'] = int(new_dic['cx1']//1)

else:
    right_lungs[f'x'] = new_dic['x2']
    right_lungs[f'y'] = new_dic['y2']
    right_lungs[f'w'] = new_dic['w2']
    right_lungs[f'h'] = new_dic['h2']
    right_lungs[f'A'] = new_dic['A2']
    right_lungs[f'cx'] = int(new_dic['cx2']//1)


y1 =left_lungs['y'] if left_lungs['y'] < right_lungs['y'] else right_lungs['y']
w =left_lungs['w'] if left_lungs['w'] > right_lungs['w'] else right_lungs['w']
h =left_lungs['h'] if left_lungs['h'] > right_lungs['h'] else right_lungs['h']
print(y,w,h)


if left_lungs['A']>right_lungs['A']:
    print('left')
    if right_lungs['cx']+(w//2) > width:
        crop_img = img[y1:y1+h, left_lungs['x']:width]
        cv2.imwrite(f'new.png',crop_img)
    else:
        crop_img = img[y1:y1+h, left_lungs['x']:right_lungs['cx']+(w//2)]
        cv2.imwrite(f'new.png',crop_img)
    " crop the images with original image size "
    # mask = np.zeros_like(img)
    # mask=cv2.rectangle(mask, (left_lungs['x'], y1), (right_lungs['cx']+(w//2), y1+ h), (255, 255, 255), -1)
    # result = cv2.bitwise_and(img, mask)

else:
    print('right')

    if left_lungs['cx']-(w//2) < 0:
        crop_img = img[y1:y1+h, 0:right_lungs['x']+w]
        cv2.imwrite(f'new.png',crop_img)
    else:
        crop_img = img[y1:y1+h, left_lungs['cx']-(w//2):right_lungs['x']+w]
        cv2.imwrite(f'new.png',crop_img)
    # mask = np.zeros_like(img)
    # mask=cv2.rectangle(mask, (left_lungs['cx']-(w//2), y1), (right_lungs['x']+w, y1+ h), (255, 255, 255), -1)
    # result = cv2.bitwise_and(img, mask)

    # print(np.amax(img))
    # print(left_lungs['cx']-(w//2))
    # print(right_lungs['x'])
    # print(w)
    # print(crop_img.size)

        
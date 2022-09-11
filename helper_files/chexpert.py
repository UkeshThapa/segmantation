
import os
import glob
import cv2
import numpy as np
import csv
import shutil

''' Extract the large component '''
# pattern = r'E:\Data\Dataset\Results\Segmentation\u5_model\vinbigdata' + "\*"
# dst_folders=r'E:\Data\Dataset\Processed_image\vinbigdata\flag_images\flag_mask_during_post_processing\\'

# for file in glob.iglob(pattern, recursive=True):    
#     file_name = os.path.basename(file)

#     img = cv2.imread(file)
#     gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#     nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)
#     sizes = stats[:, -1]
#     new_dict ={}
#     for i in range(1,nb_components):
#         new_dict[f'{i}'] = sizes[i]
#     num_component = len(new_dict)
#     try:
#         if num_component == 1:
#             print('The masks has 1 component ')
#             txt = open('vinbigdata_flag_image.txt','a')
#             shutil.copy(file, dst_folders + file_name)
#             txt.write(str(file_name)+' '+'1 component'+'\n')
#             txt.close()
#         else :
#             max_key = max(new_dict, key=new_dict.get)
#             img2 = np.zeros(output.shape)
#             img2[output==int(max_key)] = 255
#             new_dict.pop(max_key)
#             sec_max_key = max(new_dict, key=new_dict.get)
#             img3 = np.zeros(output.shape)
#             img3[output==int(sec_max_key)] = 255
#             img4=img2+img3
#         cv2.imwrite(f"E:/Data/Dataset/Processed_image/post_process/vinbigdata/{file_name}", img4, [cv2.IMWRITE_PNG_BILEVEL, 1]) 
#     except:
#             print('The masks has No component ')
#             txt = open('vinbigdata_flag_image.txt','a')
#             txt.write(str(file_name)+' '+'No component'+'\n')
#             txt.close()


''' move the images from original images of cheXpert to folder '''
# with open(os.path.join(os.path.join('vinbigdata_flag_image.txt')), "r") as f:
#     lines = f.read().splitlines()
#     print(lines)

# list_name = []
# for ii, line in enumerate(lines):

#     name ,dot,ext = line.partition(' ')
#     list_name.append(name)

# paths =r'E:\Data\Dataset\Processed_image\vinbigdata\images'+'\*'
# dst_folders = r"E:\Data\Dataset\Processed_image\vinbigdata\flag_images\flag_image_during_post_processing\\"
# for file in glob.iglob(paths, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     if file_name in list_name:
#         print(file_name)
#         shutil.move(file, dst_folders + file_name)

''' find the coordinate boundary '''

pred_mask = r'E:\Data\Dataset\Processed_image\post_process\vinbigdata\\' 
img_path =r'E:\Data\Dataset\Processed_image\vinbigdata\images'+ "\*"

for file in glob.iglob(img_path, recursive=True):    
    file_name = os.path.basename(file)
    name,dot,ext=file_name.partition('.')
    print(file_name)
    img = cv2.imread(file)
    height,width,channels = img.shape
    mask_img = cv2.imread(pred_mask+name+'.png')

    # compare the mean
    try:
        mask=cv2.imread(pred_mask+name+'.png',0)
        height ,width= mask.shape
        percentage=(cv2.countNonZero(mask)/(height*width))*100
        if percentage > 7:

            gray = cv2.cvtColor(mask_img , cv2.COLOR_BGR2GRAY)

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

                new_dic[f'x{i}'] = x
                new_dic[f'y{i}'] = y
                new_dic[f'w{i}'] = w
                new_dic[f'h{i}'] = h
                new_dic[f'A{i}'] = area
                new_dic[f'cx{i}'] = cX
                new_dic[f'cy{i}'] = cY




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
            # print(y,w,h)


            if left_lungs['A']>right_lungs['A']:
                print('left')
                if right_lungs['cx']+(w//2) > width:
                    # crop_img = img[y1:y1+h, left_lungs['x']:width]
                    data=[file_name,left_lungs['x'],y1,width,y1+h]
                    with open('vinbigdata.csv', 'a', encoding='UTF8',newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)
                else:
                    # crop_img = img[y1:y1+h, left_lungs['x']:right_lungs['cx']+(w//2)]
                    data=[file_name,left_lungs['x'],y1,right_lungs['cx']+(w//2),y1+h]
                    with open('vinbigdata.csv', 'a', encoding='UTF8',newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)

                # cv2.imwrite(f'E:/Data/cheXpert_dataset/after_crop/{file_name}',crop_img)
            else:
                print('right')
                if left_lungs['cx']-(w//2) < 0:
                    # crop_img = img[y1:y1+h, 0:right_lungs['x']+w]
                    data=[file_name,0,y1,right_lungs['x']+w,y1+h]
                    with open('vinbigdata.csv', 'a', encoding='UTF8',newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)


                else:
                    # crop_img = img[y1:y1+h, left_lungs['cx']-(w//2):right_lungs['x']+w]
                    data=[file_name,left_lungs['cx']-(w//2),y1,right_lungs['x']+w,y1+h]
                    with open('vinbigdata.csv', 'a', encoding='UTF8',newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(data)


                # cv2.imwrite(f'E:/Data/cheXpert_dataset/after_crop/{file_name}',crop_img)
        else:
            txt = open('3std_vinbigdata1.txt','a')
            txt.write(str(file_name)+'\n')
            txt.close()
    except:
            txt = open('vinbigdata_image_error1.txt','a')
            txt.write(str(file_name)+'\n')
            txt.close()
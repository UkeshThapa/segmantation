from fileinput import filename
import os
import glob
import cv2 as cv
from cv2 import imread
from cv2 import add
import numpy as np
from PIL import Image
import cv2

"""
  china dataset

"""
# pattern = r'E:\Data\downloads\chest-x-rays\ChinaSet_AllFiles\ChinaSet_Masks\mask' + "\*"

# for file in glob.iglob(pattern, recursive=True):
#     file_name = os.path.basename(file)
#     masks_img = cv.imread(file)
#     height,width,chann=masks_img.shape
#     new_image = np.zeros(( height,width))
#     masks_img = masks_img/np.amax(masks_img)

#     cv.imwrite(f"E:/Data/Process/Chain_x-ray/masks/{file_name}", new_image)





# for file in glob.iglob(left_masks, recursive=True):
#     file_name = os.path.basename(file)
#     masks_img = cv.imread(file)
#     height,width,chann=masks_img.shape
#     new_image = np.zeros(( height,width))
#     masks_img = masks_img/np.amax(masks_img)











''''
Japan dataset

'''

'''images'''
# pattern = r'E:\Data\downloads\chest-x-rays\JSRT\images\JPCN' + "\*"

# for file in glob.iglob(pattern, recursive=True):
#     file_name = os.path.basename(file)
#     name,dot,ext = file_name.partition('.')
#     im = Image.open(file)
#     im = im.convert("L")
#     im_np=np.array(im)
#     # im_np = im_np.astype(np.uint8)
#     print(np.amax(im_np),np.amin(im_np))
#     im.save(f'E:/Data/Process/japan_x-ray/images/{name}.png')

'''masks'''
# pattern = r'E:\Data\downloads\chest-x-rays\XLSor_data\data\NIH\masks' + "\*"

# for file in glob.iglob(pattern, recursive=True):    
#     file_name = os.path.basename(file)
#     name,dot,ext = file_name.partition('.')
#     im = Image.open(file)
#     im_np=np.array(im)
#     save_name = f'E:/Data/Process/NIH/masks/{name}.png'
#     cv.imwrite(save_name,im_np.astype(int), [cv.IMWRITE_PNG_BILEVEL, 1] )


''''NLM'''


# left_mask= r'E:\Data\downloads\chest-x-rays\NLM-MontgomeryCXRSet\MontgomerySet\ManualMask\leftMask'+'/*'
# # image3 = np.zeros(( 512,512))
# # right = cv.imread(right_mask)
# # image3
# # cv.imwrite('ko.png',image3)

# for file in glob.iglob(left_mask, recursive=True):
#     file_name = os.path.basename(file)
#     right_mask = r'E:\Data\downloads\chest-x-rays\NLM-MontgomeryCXRSet\MontgomerySet\ManualMask\rightMask'+f'/{file_name}'

#     left = cv.imread(file)
#     height,width,chann=left.shape
#     new_image = np.zeros(( height,width))
#     right= cv.imread(right_mask)
#     left_np=np.array(left)
#     right_np=np.array(right)
#     new_image=left_np+right_np
#     new_image[new_image==255]=1
#     images= new_image[:,:,0]
    
#     print(np.shape(images))
#     print(np.amax(images),np.amin(images))

# #     masks_img = masks_img/np.amax(masks_img)

#     cv.imwrite(f"E:/Data/Process/NLM-MontgomeryCXRSet/masks/{file_name}", images, [cv.IMWRITE_PNG_BILEVEL, 1])

# WRITE THE FILE IN THE FOLDER




    # except:
    #   img = cv2.imread(file)
    #   gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)


    #   nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray, connectivity=4)

    #   sizes = stats[:, -1]


    #   list_sizes = [sizes[i] for i in range(1,nb_components)]
    #   total_size = sum(list_sizes)
    #   list_label=[]
    #   for i in range(1, nb_components):
    #     percent_sizes=(sizes[i]/total_size)*100
    #     if percent_sizes > 8:
    #       list_label.append(i)
    #   print(list_label)
    #   print('this one has one lungs in connected object',file_name)
    #   img2 = np.zeros(output.shape)
    #   img3 = np.zeros(output.shape)
    #   img2[output == list_label[0]] = 255
      # cv2.imwrite(f"E:/Data/post-process_output_xray/Montgomery_xrays/{file_name}", img2, [cv2.IMWRITE_PNG_BILEVEL, 1])

# print(len(list_name))
# text_file1 = open("E:/Data/Process/Chain_x-ray/split_Dataset/test.txt", "w")

# for i in range(len(list_name)):

#       print(list_name[i].split('_mask')[0])
#       text_file1.write(list_name[i].split('_mask')[0]+'.png')
#       text_file1.write('\n')
# #close file
# text_file1.close()


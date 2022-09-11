import cv2
import numpy as np
import os
import json
import numpy as np

# E:/Data/X-ray_Dataset/annotations

path_to_json = 'E:/Data/X-ray_Dataset/annotations/'
count = 0
for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
  with open(path_to_json + file_name) as json_file:
    data = json.load(json_file)
    try:


        list_annotation = data['annotations']
        height = data['image']['height']
        width = data['image']['width']
        file_name = data['image']['filename']
        name ,dot,ext = file_name.partition('.')

        lungs_pixel = []
        for i in range (len(list_annotation)):
            lungs_finder = data['annotations'][i]['name']
            if lungs_finder == 'Lung':

                pixel_pts = data['annotations'][i]['polygon']["path"]
                lungs_pixel.append(pixel_pts)
        
        formate_pixel_pts_left=[[i['x'],i['y']] for i in lungs_pixel[0]]
        formate_pixel_pts_right=[[i['x'],i['y']] for i in lungs_pixel[1]]

        final_pts_left = np.array(formate_pixel_pts_left, np.int32)
        final_pts_right = np.array(formate_pixel_pts_right, np.int32)

        image3 = np.zeros(( height,width))

        cv2.fillPoly(image3, pts = [final_pts_left], color =(255,255,255))
        cv2.fillPoly(image3, pts = [final_pts_right], color =(255,255,255))
        image3 = image3/np.amax(image3)
        print(np.amax(image3))
        # image3.dtype='uint8'
        # print(np.amax(image3))
        cv2.imwrite(f"E:/Data/X-ray_Dataset/new_masks/mask_{name}.png", image3)



    except:


        
            list_annotation = data['annotations']
            height = data['image']['height']
            width = data['image']['width']
            file_name = data['image']['filename']
            name ,dot,ext = file_name.partition('.')

            lungs_pixel = []
            for i in range (len(list_annotation)):
                lungs_finder = data['annotations'][i]['name']
                if lungs_finder == 'Lung':

                    pixel_pts = data['annotations'][i]['polygon']["path"]
                    lungs_pixel.append(pixel_pts)
            
            if len(lungs_pixel) == 1:
                print(f'{file_name}')
                formate_pixel_pts=[[i['x'],i['y']] for i in lungs_pixel[0]]


                final_pts = np.array(formate_pixel_pts, np.int32)

                image3 = np.zeros(( height,width))

                cv2.fillPoly(image3, pts = [final_pts], color =(255,255,255))
                image3 = image3/np.amax(image3)
                print(np.amax(image3))
                # image3.dtype='uint8'
                # print(np.amax(image3))
                cv2.imwrite(f"E:/Data/X-ray_Dataset/1/mask_{name}.png", image3)


        
                # file_name = data['image']['filename']
                # CT_name=data['annotations'][0]['name']
                # try:
                #           CT_name=data['annotations'][2]['name']
                #           print(f"{file_name} {CT_name}")
                # except:
                #     print(f"{file_name} {CT_name}")



  count=count+1
print(count)
      

# images = cv2.imread('D:\\Maya.Ai Reseacrh Center\\x-ray\\Dataset\\VOC2012\\SegmentationObject\\2007_000129.png')
# print(np.shape(images))

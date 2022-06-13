import os
import cv2 
import numpy as np
import glob
'''

find the two largest connected components

'''
# pattern = r'E:\Results\Segmentation\cheXpert_dataset' + "\*"
# no_pred=[]
# for file in glob.iglob(pattern, recursive=True):    
#     file_name = os.path.basename(file)

#     img = cv2.imread('00001809_004.png')
#     gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
#     nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)
#     sizes = stats[:, -1]
#     new_dict ={}
#     for i in range(1,nb_components):
#         new_dict[f'{i}'] = sizes[i]
#     max_key = max(new_dict, key=new_dict.get)
#     img2 = np.zeros(output.shape)
#     img2[output==int(max_key)] = 255
#     new_dict.pop(max_key)
#     sec_max_key = max(new_dict, key=new_dict.get)
#     img3 = np.zeros(output.shape)
#     img3[output==int(sec_max_key)] = 255
#     img4=img2+img3
#     cv2.imwrite(f"00001809_004.png", img4, [cv2.IMWRITE_PNG_BILEVEL, 1]) 
#    except:
#         no_pred.append(file_name)
#         print('not prd',file_name)
        
# text_file1 = open("issue.txt", "w")

# for i in range(len(no_pred)):
#     text_file1.write(no_pred[i])
#     text_file1.write('\n')
#     print(i)
# #close file
# text_file1.close()

''' 



'''



img = cv2.imread('00025444_002.png')
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
nb_components, output, stats, _ = cv2.connectedComponentsWithStats(gray)
sizes = stats[:, -1]
new_dict ={}
for i in range(1,nb_components):
    new_dict[f'{i}'] = sizes[i]
num_component = len(new_dict)
if num_component == 1:
    print('The masks has 1 component ') 
else :
    max_key = max(new_dict, key=new_dict.get)
    img2 = np.zeros(output.shape)
    img2[output==int(max_key)] = 255
    new_dict.pop(max_key)
    sec_max_key = max(new_dict, key=new_dict.get)
    img3 = np.zeros(output.shape)
    img3[output==int(sec_max_key)] = 255
    img4=img2+img3
    cv2.imwrite(f"00025444_002.png", img4, [cv2.IMWRITE_PNG_BILEVEL, 1])

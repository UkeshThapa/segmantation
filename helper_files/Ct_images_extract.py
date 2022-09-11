import json
import os
import glob
import shutil

# path_to_json = 'D:/Maya.Ai Reseacrh Center/x-ray/Dataset/covid-19-chest-x-ray-dataset/releases/all-images/annotations/'
# count=0
# list_CT_images = []
# for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
#   with open(path_to_json + file_name) as json_file:
#     data = json.load(json_file)
#     CT_name=data['annotations'][0]['name']
#     if CT_name == 'CT':
#         file_name = data['image']['filename']
#         list_CT_images.append(file_name)
#         count=count+1

# print(count)
# print(list_CT_images)




src_folder = r"E:\Data\processed_image\Orignal_4_Dataset\Chain_x-ray\masks"
dst= r"E:\Data\processed_image\Orignal_4_Dataset\Chain_x-ray\images"+'\*'
save_folder =r'E:\Data\processed_image\Orignal_4_Dataset\Chain_x-ray\dep_new_img\\' 
pattern = src_folder + "\*"
list_mask=[]
for file in glob.iglob(pattern, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    name,dot,ext=file_name.partition('_mask')
    list_mask.append(name)


for file in glob.iglob(dst, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    name,dot,ext=file_name.partition('.')

    if name in list_mask:
        print(name)
        shutil.move(file, save_folder + file_name)
        print('Moved:', file)
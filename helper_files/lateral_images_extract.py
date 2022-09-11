import json
import os
import glob
import shutil

path_to_json = 'D:/Maya.Ai Reseacrh Center/x-ray/Dataset/covid-19-chest-x-ray-dataset/releases/all-images/annotations/'
count=0
list_CT_images = []
for file_name in [file for file in os.listdir(path_to_json) if file.endswith('.json')]:
  with open(path_to_json + file_name) as json_file:
    data = json.load(json_file)
    try:
      CT_name=data['annotations'][2]['name']
      if CT_name == "View/Lateral":
            file_name = data['image']['filename']
            list_CT_images.append(file_name)
            count=count+1

    except:
      print('Not lateral')

print(count)
print(list_CT_images)


src_folder = r"D:\Maya.Ai Reseacrh Center\x-ray\Dataset\covid-19-chest-x-ray-dataset\images"
dst_folder = r"D:\Maya.Ai Reseacrh Center\x-ray\Dataset\covid-19-chest-x-ray-dataset\lateral_images\\"

pattern = src_folder + "\*"

for file in glob.iglob(pattern, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    if file_name in list_CT_images:
        shutil.move(file, dst_folder + file_name)
        print('Moved:', file)






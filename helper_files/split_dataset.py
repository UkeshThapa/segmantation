
import random
import glob
import os
import pandas as pd
import shutil
# df= pd.read_csv('new_csv.csv')
# print(df['No Finding'].value_counts())
# print(df['Enlarged Cardiomediastinum'].value_counts())
# print(df['Cardiomegaly'].value_counts())
# print( df['Lung Lesion'].value_counts())
# print(df['Edema'].value_counts())
# print(df['Consolidation'].value_counts())
# print(df['Pneumonia'].value_counts())
# print(df['Atelectasis'].value_counts())
# print(df['Pleural Effusion'].value_counts())
# print(df['Fracture'].value_counts())
# print(df['Support Devices'].value_counts())
# print(df['Lung Opacity'].value_counts())
# print(df['Pneumothorax'].value_counts())
# print(df['Pleural Other'].value_counts())



# df['Finding_Labels']=df['Finding Labels']
# df=df[['Image Index','Finding_Labels']]
# df.loc[df.Finding_Labels != 'No Finding', 'Finding_Labels'] = 'Abnormal'
# df.loc[df.Finding_Labels == 'No Finding', 'Finding_Labels'] = 'Normal'
# print(df['Finding_Labels'].value_counts())
# df.to_csv('classification_lungs.csv')
# df.drop(df[df["Frontal/Lateral"] == 'Lateral'].index, inplace = True)
# df.to_csv('new_csv.csv')
# df1 = df.loc[df["Frontal/Lateral"] =='Frontal' ]
# new_df=df1[['Path',"Frontal/Lateral"]]
# print(len(df.index))
# print(df.head(5))
# for i in range (len(df.index)):

#     path=df["Path"].values[i]
#     print(path)
#     others,mid,name=path.partition('valid/')
#     nam=name.replace('/','_')
#     shutil.move('E:/Data/CheXpert-v1.0-small/'+path, 'E:/Data/CheXpert-v1.0-small/valid/' + nam)
#     print(nam)



    







# random.seed(100)


# path =r'E:\Data\processed_image\post-process_output_xray\NIH_large_dataset'+'\*'
# list_name_img = []
# for file in glob.iglob(path, recursive=True):
# #     # extract file name form file path
#     file_name = os.path.basename(file)
#     list_name_img.append(file_name)

# pattern = r'E:\Results\Segmentation\NIH_dataset' + "\*"
# # dst_folder = r"E:\Data\NIH_dataset\flag_seg\\"
# list_name_images =[]
# for file in glob.iglob(pattern, recursive=True):
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     # name , extension = os.path.splitext(file_name)
#     if file_name not in list_name_img:
#         list_name_images.append(file_name)
#         # shutil.move(file, dst_folder + file_name)
#         # print('Moved:', file)

# print('''

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!/n

# ''')


paths =r'E:\Data\Dataset\Processed_image\vinbigdata\images'+'\*'
# dst_folders = r"E:\Data\NIH_dataset\flaged_image\\"
for file in glob.iglob(paths, recursive=True):
    # extract file name form file path
    file_name = os.path.basename(file)
    txt=open('test.txt','a')
    txt.write(file_name+'\n')
    txt.close()

#     if file_name in list_name_images:
#         shutil.move(file, dst_folders + file_name)
#         print('Moved:', file)


# print(len(list_name_images))

# random_train_val = random.sample(list_name_images, int(len(list_name_images)*0.8))

# random_val = random.sample(random_train_val, int(len(random_train_val)*0.2))
# print(len(random_val))
# random_train = []
# for i in range(len(random_train_val)):
#     if random_train_val[i] not in random_val:

#         random_train.append(random_train_val[i])

# print(len(random_train))

# random_test=[]
# for i in range(len(list_name_images)):
#     if list_name_images[i] not in random_train_val:

#         random_test.append(list_name_images[i])

# print(len(random_test))



# # # WRITE THE FILE IN THE FOLDER
# text_file1 = open("E:/Data/CheXpert-v1.0-small/split_Dataset/test.txt", "w")

# for i in range(len(list_name_images)):

 
#     text_file1.write(list_name_images[i])
#     text_file1.write('\n')
# #close file
# text_file1.close()

# text_file2 = open("E:/Data/processed_image/Orignal_4_Dataset/NLM-MontgomeryCXRSet/val.txt", "w")

# for i in range(len(random_val)):

 
#     text_file2.write(random_val[i])
#     text_file2.write('\n')
# #close file
# text_file2.close()

# text_file3 = open("E:/Data/processed_image/Orignal_4_Dataset/NLM-MontgomeryCXRSet/test.txt", "w")

# for i in range(len(random_test)):

 
#     text_file3.write(random_test[i])
#     text_file3.write('\n')
# #close file
# text_file3.close()


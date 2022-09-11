import random
import glob
import os
import pandas as pd
import shutil
    
'''
In this dataset is split into train-val and test 
by the ratio of 80:20 
and again train-val is split in the 80:20
with seed of 100
'''

random.seed(100)

# list the total image 
paths =r'E:\Data\Dataset\Processed_image\lungs-segmentation-dataset\darwinlungs\images'+'\*'
list_name_images =[]
for file in glob.iglob(paths, recursive=True):
    file_name = os.path.basename(file)
    list_name_images.append(file_name)

new_data = random.sample(list_name_images, 100)

# split to train and validation
random_train_val = random.sample(new_data, int(len(new_data)*0.8))

# list of validation dataset
random_val = random.sample(random_train_val, int(len(random_train_val)*0.2))

# list of train dataset
random_train = []
for i in range(len(random_train_val)):

    # check the name list for train not in val list
    if random_train_val[i] not in random_val:
        random_train.append(random_train_val[i])

# list the test dataset
random_test=[]
for i in range(len(new_data)):
    if new_data[i] not in random_train_val:
        random_test.append(new_data[i])


# WRITE THE FILE IN THE FOLDER

# train 
text_file1 = open("train.txt", "w")

for i in range(len(random_train)):
    text_file1.write(random_train[i])
    text_file1.write('\n')
#close file
text_file1.close()


# Validation
text_file2 = open("val.txt", "w")
for i in range(len(random_val)):
    text_file2.write(random_val[i])
    text_file2.write('\n')
#close file
text_file2.close()

# test
text_file3 = open("test.txt", "w")
for i in range(len(random_test)): 
    text_file3.write(random_test[i])
    text_file3.write('\n')
#close file
text_file3.close()


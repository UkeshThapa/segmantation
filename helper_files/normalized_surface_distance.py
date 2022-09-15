import os
import csv
import cv2 
import numpy as np
from medpy.metric.binary import __surface_distances



def normalized_surface_dice(a: np.ndarray, b: np.ndarray, threshold: float, spacing: tuple = None, connectivity=1):

    """
    This implementation differs from the official surface dice implementation! These two are not comparable!!!!!
    The normalized surface dice is symmetric, so it should not matter whether a or b is the reference image
    This implementation natively supports 2D and 3D images. Whether other dimensions are supported depends on the
    __surface_distances implementation in medpy
    :param a: image 1, must have the same shape as b
    :param b: image 2, must have the same shape as a
    :param threshold: distances below this threshold will be counted as true positives. Threshold is in mm, not voxels!
    (if spacing = (1, 1(, 1)) then one voxel=1mm so the threshold is effectively in voxels)
    must be a tuple of len dimension(a)
    :param spacing: how many mm is one voxel in reality? Can be left at None, we then assume an isotropic spacing of 1mm
    :param connectivity: see scipy.ndimage.generate_binary_structure for more information. I suggest you leave that
    one alone
    :return:
    """
    assert all([i == j for i, j in zip(a.shape, b.shape)]), "a and b must have the same shape. a.shape= %s, " \
                                                            "b.shape= %s" % (str(a.shape), str(b.shape))
    if spacing is None:
        spacing = tuple([1 for _ in range(len(a.shape))])
    a_to_b = __surface_distances(a, b, spacing, connectivity)
    b_to_a = __surface_distances(b, a, spacing, connectivity)
    numel_a = len(a_to_b)
    numel_b = len(b_to_a)
    tp_a = np.sum(a_to_b <= threshold) / numel_a
    tp_b = np.sum(b_to_a <= threshold) / numel_b
    fp = np.sum(a_to_b > threshold) / numel_a
    fn = np.sum(b_to_a > threshold) / numel_b
    dc = (tp_a + tp_b) / (tp_a + tp_b + fp + fn + 1e-8)  # 1e-8 just so that we don't get div by 0
    return dc



def mid_x_coordinate(value):
    return value//2


def crop_image(img,x,h,w,part):
    
    if part == 'left':
        crop_img = img[0:h, x:w]        
        gray = cv2.cvtColor(crop_img , cv2.COLOR_BGR2GRAY)
        return gray

    
    elif part == 'right':
        crop_img = img[0:h, 0:x]
        gray = cv2.cvtColor(crop_img , cv2.COLOR_BGR2GRAY)
        return gray

    elif part == 'whole':
        gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
        return gray

# def save_image(img):
#     cv2.imwrite('crop_gt.png',img, [cv2.IMWRITE_PNG_BILEVEL, 1])

if __name__ == "__main__":

    model=["nih_xrays_dataset","montgomery_xrays_dataset","japan_xrays_dataset","china_xrays_dataset","darwinlungs"]
    dataset_names=["nih_xrays_dataset","montgomery_xrays_dataset","japan_xrays_dataset","china_xrays_dataset","darwinlungs"]
    DSC = input('Choose Dice score options..\n1. Whole masks Dice score\n2. Right lung masks Dice score\n3. Left lung masks Dice score\n')
    
    # Creating the csv file
    header = ["Dataset","nih_xrays_dataset","montgomery_xrays_dataset","japan_xrays_dataset","china_xrays_dataset","darwinlungs"]

    if DSC == '1':
        name_part = 'whole'
    elif DSC == '2':
        name_part = 'right'
    elif DSC == '3':
        name_part = 'left'


    with open(f'./results/NSD_{name_part}.csv', 'w', encoding='UTF8',newline='') as f:
    # create the csv writer
        writer = csv.writer(f)
        writer.writerow(header)
        
    for i in range(len(model)):
        nam = model[i]
        list_name=[]
        list_name.append(model[i])
        for sub_name in dataset_names:
            prd_path = f'E:\\Data\\Dataset\\Results\\Segmentation\\segmentation_513\\segmentation_pretrain\\{nam}_model\\{sub_name}\\'
            gt_path = f'E:\\Data\\Dataset\\Processed_image\\lungs-segmentation-dataset\\{sub_name}\\masks\\'
            prd_list = os.listdir(prd_path)
            mean_dice = []
            for fname in prd_list:
                name,dot,ext=fname.partition('.')
                try:

                    gt = cv2.imread(gt_path+name+'_mask.png')

                    height, width, channel = gt.shape
                    x_center = mid_x_coordinate(width) 
                    pred = cv2.imread(prd_path+fname)

                    gt_crop_img = np.array(crop_image(gt,x_center,height,width,name_part))  
                    pred_crop_img = np.array(crop_image(pred,x_center,height,width,name_part))   

                    mean_dice.append(normalized_surface_dice(pred_crop_img,gt_crop_img,1))
                    print("NSD is :",fname,normalized_surface_dice(pred_crop_img,gt_crop_img,1))
            

                except:
                    gt = cv2.imread(gt_path+name+'.png')

                    height, width, channel = gt.shape
                    x_center = mid_x_coordinate(width) 
                    pred = cv2.imread(prd_path+fname)

                    gt_crop_img = np.array(crop_image(gt,x_center,height,width,name_part))  
                    pred_crop_img = np.array(crop_image(pred,x_center,height,width,name_part))   
                    mean_dice.append(normalized_surface_dice(pred_crop_img,gt_crop_img,1))
                    print("NSD is :",fname,normalized_surface_dice(pred_crop_img,gt_crop_img,1))
            
            m_dice = sum(mean_dice)/len(mean_dice)
            list_name.append(m_dice)
            print(f'---> average NSD {sub_name}',m_dice)
        data=[list_name[0],list_name[1],list_name[2],list_name[3],list_name[4],list_name[5]]    
        with open(f'./results/NSD_{name_part}.csv', 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

import os
import csv
import cv2 
import numpy as np

def compute_dice_coefficient(mask_gt, mask_pred):
  """Computes soerensen-dice coefficient.

  compute the soerensen-dice coefficient between the ground truth mask `mask_gt`
  and the predicted mask `mask_pred`.

  Args:
    mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
    mask_pred: 3-dim Numpy array of type bool. The predicted mask.

  Returns:
    the dice coeffcient as float. If both masks are empty, the result is NaN.
  """
  volume_sum = mask_gt.sum() + mask_pred.sum()
  if volume_sum == 0:
    return np.NaN
  volume_intersect = (mask_gt & mask_pred).sum()
  return 2*volume_intersect / volume_sum 


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
    DSC = input('Choose Dice score options..\n1. Whole masks Dice score\n2. Right lung masks Dice score\n3. Left lung masks Dice score')
    
    # Creating the csv file
    header = ["Dataset","nih_xrays_dataset","montgomery_xrays_dataset","japan_xrays_dataset","china_xrays_dataset","darwinlungs"]

    if DSC == '1':
        name_part = 'whole'
    elif DSC == '2':
        name_part = 'right'
    elif DSC == '3':
        name_part = 'left'


    with open(f'dice_score_{name_part}.csv', 'w', encoding='UTF8',newline='') as f:
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

                    mean_dice.append(compute_dice_coefficient(gt_crop_img,pred_crop_img))
                    print("dice score is :",fname,compute_dice_coefficient(gt_crop_img,pred_crop_img))
            

                except:
                    gt = cv2.imread(gt_path+name+'.png')

                    height, width, channel = gt.shape
                    x_center = mid_x_coordinate(width) 
                    pred = cv2.imread(prd_path+fname)

                    gt_crop_img = np.array(crop_image(gt,x_center,height,width,name_part))  
                    pred_crop_img = np.array(crop_image(pred,x_center,height,width,name_part))   
                    mean_dice.append(compute_dice_coefficient(gt_crop_img,pred_crop_img))
                    print("dice score is :",fname,compute_dice_coefficient(gt_crop_img,pred_crop_img))
            
            m_dice = sum(mean_dice)/len(mean_dice)
            list_name.append(m_dice)
            print(f'---> average dice score{sub_name}',m_dice)
        data=[list_name[0],list_name[1],list_name[2],list_name[3],list_name[4],list_name[5]]    
        with open(f'dice_score_{name_part}.csv', 'a', encoding='UTF8',newline='') as f:
            writer = csv.writer(f)
            writer.writerow(data)

import argparse
from tkinter import image_names 
import torch
import numpy as np
import os
import torchvision
import random
from modeling.deeplab import *
import cv2 as cv
from dataloaders import make_data_loader
from tqdm import tqdm    
from PIL import Image
from modeling.sync_batchnorm.replicate import patch_replication_callback
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
from scipy.special import softmax
from dataloaders import custom_transforms as tr
from modeling.fcn import *

def save_output_images(LOAD_PATH_MODEL, model, test_loader, SAVE_PATH_MODEL_OUTPUT,gt_paths,dataset_name):
    '''Saves the prediction image from segmentation model in local disk. Supports only batch size 1 now.
    Args:
        LOAD_PATH_MODEL: Path where model weights are saved
        test_loader: dataloader
        SAVE_PATH_MODEL_OUTPUT: save path for segmentation outputs

       
    '''

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    checkpoint = torch.load(LOAD_PATH_MODEL)

    start_epoch = checkpoint['epoch']
    
    model.load_state_dict(checkpoint['state_dict'])


    model.eval()

    tbar = tqdm(test_loader, desc='\r')

    for i, sample in enumerate(tbar):

            image = sample['image'] 
            image_names = sample['image_name']  
            # print('sss',image_names[0],len(image_names))
            image = image.to(device)

            with torch.no_grad():
                output = model(image)
                output = output['out']
                output = output.data.cpu().numpy()  
            pred = output[0,:,:,:]
            pred = softmax(pred, axis=0)
            pred = np.argmax(pred, axis= 0)

            # pred = Image.fromarray((pred))
            name ,_,_ = image_names[0].partition('.')

# UPDATED BY UKESH

            if dataset_name == 'japan_xrays_dataset' or dataset_name == 'montgomery_xrays_dataset': 
                img = cv.imread(f"{gt_paths}\\masks\\{name}.png")
            else:
                img = cv.imread(f"{gt_paths}\\masks\\{name}_mask.png")
            h,w,c = img.shape 

            # pred.save()
            cv.imwrite(SAVE_PATH_MODEL_OUTPUT+'\\' + name+'.png' , pred, [cv.IMWRITE_PNG_BILEVEL, 1])
            
            preds = cv.imread(f"{SAVE_PATH_MODEL_OUTPUT}\\{name}.png") 
            resized = cv.resize(preds, (w, h), 0, 0, interpolation = cv.INTER_NEAREST)
            gray_image = cv.cvtColor(resized, cv.COLOR_BGR2GRAY)
            cv.imwrite(SAVE_PATH_MODEL_OUTPUT+'\\' + name+'.png' ,gray_image,[cv.IMWRITE_PNG_BILEVEL, 1])
            # print((SAVE_PATH_MODEL_OUTPUT +'\\'  



if __name__ == '__main__':
    list_model = ['japan_xrays_dataset']
    list_dataset = ['china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','nih_xrays_dataset','darwinlungs']
    
    for name in list_model:
        print(f'Loading Dataset model --> {name}')
        
        LOAD_PATH_MODEL = f'E:\\projects\\X-ray\\Deeplab-Xception-Lungs-Segmentation-master\\run\\{name}\\deeplab-resnet\\model_best.pth.tar' 
                
        for datasets in list_dataset:
            print(f'Testing on Dataset --> {datasets}')
            parser = argparse.ArgumentParser(description="Inference For Best Model")
            parser.add_argument('--dataset', type=str, default=datasets, choices=['pascal', 'camus','darwinlungs','china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','nih_xrays_dataset','u4_dataset','u5_dataset','NIH_dataset','cheXpert_dataset'])
            parser.add_argument('--batch_size', type=int, default=1)
            parser.add_argument('--no-cuda', action='store_true', default=
                                False, help='disables CUDA training')
            parser.add_argument('--gpu-ids', type=str, default='0',
                                help='use which gpu to train, must be a \
                                comma-separated list of integers only (default=0)')
            # # added by uk
            parser.add_argument('--size', type=int, default=513,
                                help='image size')
            args = parser.parse_args()
            kwargs = {'num_workers': 4, 'pin_memory': True}
            _, _,test_loader, nclass = make_data_loader(args, **kwargs)

            SAVE_BASE_PATH = f'E:\\Data\\Dataset\\Results\\Segmentation\\fcn_pretrain\\' 
            if os.path.isdir(SAVE_BASE_PATH+name+'_model') is False:         
                os.mkdir(SAVE_BASE_PATH+name+'_model')

            paths = SAVE_BASE_PATH+name+'_model\\'

            if os.path.isdir(paths+args.dataset) is False:
                os.mkdir(paths+args.dataset)

            # model = DeepLab(num_classes=2,
            #                     backbone='resnet',
            #                     output_stride=16,
            #                     sync_bn=False,
            #                     freeze_bn=False)
            model = FCNResNet101(2,512) #model(classes,size)
            



            args.cuda = not args.no_cuda and torch.cuda.is_available()
            if args.cuda:
                try:
                    args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
                except ValueError:
                    raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
            n_classes = 2
        
            gt_path_model = f"E:\\Data\\Dataset\\Processed_image\\lungs-segmentation-dataset\\{datasets}\\"
            save_output_images(LOAD_PATH_MODEL,model,test_loader,os.path.join(paths,args.dataset),gt_path_model,datasets)



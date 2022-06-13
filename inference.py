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

def normalize_img(img):
   
      mean = (0.485, 0.456, 0.406)
      std = (0.229, 0.224, 0.225)
      img /= 255.0
      img -= mean
      img /= std
      return img

def transform_test(sample):

    composed_transforms = transforms.Compose([
        # tr.FixScaleCrop(crop_size=self.args.crop_size),
        tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        tr.ToTensor()])

    return composed_transforms(sample)


def save_output_images(LOAD_PATH_MODEL, model, test_loader, SAVE_PATH_MODEL_OUTPUT):
    '''Saves the prediction image from segmentation model in local disk. Supports only batch size 1 now.
    Args:
        LOAD_PATH_MODEL: Path where model weights are saved
        test_loader: dataloader
        SAVE_PATH_MODEL_OUTPUT: save path for segmentation outputs

       
    '''
    
    model = torch.nn.DataParallel(model, device_ids=[0])
    patch_replication_callback(model)
    model = model.cuda()

    checkpoint = torch.load(LOAD_PATH_MODEL)
    start_epoch = checkpoint['epoch']
    # print (start_epoch)
    
    model.module.load_state_dict(checkpoint['state_dict'])


    model.eval()


    tbar = tqdm(test_loader, desc='\r')

    for i, sample in enumerate(tbar):

            image = sample['image'] 
            image_names = sample['image_name']  
            # print('sss',image_names[0],len(image_names))
            image = image.cuda()

            with torch.no_grad():
                output = model(image)
                output = output.data.cpu().numpy()  
            pred = output[0,:,:,:]
            pred = softmax(pred, axis=0)
            pred = np.argmax(pred, axis= 0)

            # pred = Image.fromarray((pred))
            name ,_,_ = image_names[0].partition('.')
            # pred.save()
            cv.imwrite(SAVE_PATH_MODEL_OUTPUT+'\\' + name+'.png' , pred, [cv.IMWRITE_PNG_BILEVEL, 1])

            # print((SAVE_PATH_MODEL_OUTPUT +'\\'  



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Inference For Best Model")
    parser.add_argument('--dataset', type=str, default='montgomery_xrays_dataset', choices=['pascal', 'camus','darwinlungs','china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','nih_xrays_dataset','u4_dataset','u5_dataset','NIH_dataset','cheXpert_dataset'])
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')

    args = parser.parse_args()
    kwargs = {'num_workers': 4, 'pin_memory': True}
    _, _,test_loader, nclass = make_data_loader(args, **kwargs)
    LOAD_PATH_MODEL = 'E:\\projects\\X-ray\\Deeplab-Xception-Lungs-Segmentation-master\\run\\china_xrays_dataset\\deeplab-resnet\\model_best.pth.tar' 


    SAVE_BASE_PATH = r'E:\Data\Dataset\Results\Segmentation\china_xrays_dataset_model/' 
    if os.path.isdir(SAVE_BASE_PATH+args.dataset) is False:
        os.mkdir(SAVE_BASE_PATH+args.dataset)

    model = DeepLab(num_classes=2,
                        backbone='resnet',
                        output_stride=16,
                        sync_bn=False,
                        freeze_bn=False)

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')
    n_classes = 2
  
 
    save_output_images(LOAD_PATH_MODEL,model,test_loader,os.path.join(SAVE_BASE_PATH,args.dataset))


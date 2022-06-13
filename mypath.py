class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return 'E:/Data/pascal/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'

            # Drawin X-ray datasets file path
        elif dataset == 'darwinlungs':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/Darwin/'
  
#   Chain  Xray file path
        elif dataset == 'china_xrays_dataset':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/china_xrays_dataset/'

            #  Japan  Xray file path
        elif dataset == 'japan_xrays_dataset':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/japan_xrays_dataset/'

            #  Montgomery  Xray file path
        elif dataset == 'montgomery_xrays_dataset':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/montgomery_xrays_dataset'


        elif dataset == 'nih_xrays_dataset':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/nih_xrays_dataset/'

        # combine datasets of China,Japan,Montgomery,NIh
        elif dataset == 'u4_dataset':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/U4/'

        # combine datasets of China,Japan,Montgomery,NIh
        elif dataset == 'u5_dataset':
            return 'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/U5/'
        # large NIH dataset
        elif dataset == 'NIH_dataset':
            return 'E:/Data/NIH_dataset/'
        # 
        elif dataset == 'cheXpert_dataset':
            return 'E:/Data/Dataset/Processed_image/vinbigdata/'

        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

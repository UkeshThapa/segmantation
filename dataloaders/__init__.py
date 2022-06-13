from dataloaders.datasets import cityscapes, combine_dbs, pascal, sbd ,darwin_lungs,japan_xray,china_xray,Montgomery_xray,NIH_xray,u4_xray,u5_xray,NIH_large_dataset,CheXpert_xray
from torch.utils.data import DataLoader

def make_data_loader(args, **kwargs):

    if args.dataset == 'pascal':
        train_set = pascal.VOCSegmentation(args, split='train')
        val_set = pascal.VOCSegmentation(args, split='val')
        if args.use_sbd:
            sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
            train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])

        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = None

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'cityscapes':
        train_set = cityscapes.CityscapesSegmentation(args, split='train')
        val_set = cityscapes.CityscapesSegmentation(args, split='val')
        test_set = cityscapes.CityscapesSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'darwinlungs':
        train_set = darwin_lungs.LungSegmentation(args, split='train')
        val_set = darwin_lungs.LungSegmentation(args, split='val')
        test_set = darwin_lungs.LungSegmentation(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class

# Training with combine Japan,china,montgomery,NIH

    elif args.dataset == 'u4_dataset':
        train_set = u4_xray.U4_dataset(args, split='train')
        val_set = u4_xray.U4_dataset(args, split='val')
        test_set = u4_xray.U4_dataset(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class

# Training with combine darwin,Japan,china,montgomery,NIH

    elif args.dataset == 'u5_dataset':
        train_set = u5_xray.U5_dataset(args, split='train')
        val_set = u5_xray.U5_dataset(args, split='val')
        test_set = u5_xray.U5_dataset(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class
  
  
    elif args.dataset == 'china_xrays_dataset':
        train_set = china_xray.China_xrays(args, split='train')
        val_set = china_xray.China_xrays(args, split='val')
        test_set = china_xray.China_xrays(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)
        
        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'japan_xrays_dataset':
        train_set = japan_xray.Japan_xrays(args, split='train')
        val_set = japan_xray.Japan_xrays(args, split='val')
        test_set = japan_xray.Japan_xrays(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class

    elif args.dataset == 'montgomery_xrays_dataset':
        train_set = Montgomery_xray.Montgomery_xrays(args, split='train')
        val_set = Montgomery_xray.Montgomery_xrays(args, split='val')
        test_set = Montgomery_xray.Montgomery_xrays(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class
   
    elif args.dataset == 'nih_xrays_dataset':
        train_set = NIH_xray.NIH_xrays(args, split='train')
        val_set = NIH_xray.NIH_xrays(args, split='val')
        test_set = NIH_xray.NIH_xrays(args, split='test')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return train_loader, val_loader, test_loader, num_class

    # large NIH dataset test
    elif args.dataset == 'NIH_dataset':
        train_set = NIH_large_dataset.NIH_xray(args, split='train')
        val_set = NIH_large_dataset.NIH_xray(args, split='val')
        test_set = NIH_large_dataset.NIH_xray(args, split='test')
        num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        # )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return test_loader, num_class
        
        #'cheXpert_dataset'
    elif args.dataset == 'cheXpert_dataset':
        train_set = CheXpert_xray.cheXpert_xray(args, split='train')
        # val_set = CheXpert_xray.cheXpert_xray(args, split='val')
        test_set = CheXpert_xray.cheXpert_xray(args, split='test')
        num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs,drop_last=True)
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs,
        # )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs,drop_last=True)

        return test_loader, num_class
  
    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError


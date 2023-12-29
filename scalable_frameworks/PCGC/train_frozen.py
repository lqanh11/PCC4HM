import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
from data_loader_h5 import ModelNetH5_voxelize_all, make_data_loader_minkowski
from pcc_model_scalable import PCCModel, PCCModel_Classification, PCCModel_Classification_Split, PCCModel_Classification_Compress
from classification_model import MinkowskiPointNet
from trainer_frozen import Trainer
import random
import sklearn.metrics as metrics

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_split_path", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/')
    parser.add_argument("--resolution", type=int, default=128, help='resolution for resampled PC')
    
    parser.add_argument("--alpha", type=float, default=16000., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--split_channel", type=int, default=4, help="number of channel for base layer.")

    # parser.add_argument("--init_ckpt", default='/media/avitech/Data/quocanhle/PointCloud/logs/Scalable_Cls/20231219_modelnet10_FIXreconstruction_TRANcls_resolution128_split_4_000/ckpts/epoch_14.pth')
    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, resolution, alpha, beta, split, lr, check_time):
        
        while os.path.exists(logdir):
            logdir = '{}_{:03d}'.format(logdir[:-4], int(logdir[-3:])+1)
            
        ckptdir = os.path.join(logdir, 'ckpts')
        
        print(logdir)
        
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.split=split
        self.lr = lr
        self.check_time = check_time
        self.resolution = resolution


def get_file_dirs(root_path):

    train_file_dirs = []
    test_file_dirs = []

    ## ModelNet10
    for filename in os.listdir(os.path.join(root_path, 'train')):
        train_file_dirs.append(os.path.abspath(os.path.join(root_path, 'train', filename)))
    for filename in os.listdir(os.path.join(root_path, 'test')):
        test_file_dirs.append(os.path.abspath(os.path.join(root_path, 'test', filename)))

    print('Train: ', len(train_file_dirs), 
        'Test: ', len(test_file_dirs))
    
    return {'Train':train_file_dirs[:], 
            'Test':test_file_dirs[:]}

if __name__ == '__main__':
    # log
    args = parse_args()
 
    prefix = f'20231219_modelnet10_FIXreconstruction_TRANcls_resolution{args.resolution}_split_{args.split_channel}_000'
    training_config = TrainingConfig(
                            logdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/Scalable_Cls/', prefix), 
                            ckptdir='', 
                            init_ckpt=args.init_ckpt, 
                            resolution=args.resolution,
                            alpha=args.alpha, 
                            beta=args.beta, 
                            split=args.split_channel,
                            lr=args.lr, 
                            check_time=args.check_time)    
    
    # Init model
    # model = PCCModel_Classification_Split(split_channel=args.split_channel)
    model = PCCModel_Classification_Compress()
    
    ## load pre-trained model
    model_dict = model.state_dict()
    model_compression = PCCModel()
    ckpt_compression = torch.load("/media/avitech/Data/quocanhle/PointCloud/compression_frameworks/PCGC/PCGCv2/logs_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth")
    model_compression.load_state_dict(ckpt_compression['model'])
    model_compression_dict = model_compression.state_dict()

    # model_classification = MinkowskiPointNet(in_channel=3, out_channel=10, embedding_channel=1024)
    # ckpt_cls = torch.load("/media/avitech/Data/quocanhle/PointCloud/logs/Mink_classification/classification_modelnet10_voxelize/1024/minkpointnet_voxelized_128/modelnet_minkpointnet.pth")
    # model_classification.load_state_dict(ckpt_cls['state_dict'])
    # model_classification_dict = model_classification.state_dict()

    processed_dict = {}

    for k in model_dict.keys(): 
        decomposed_key = k.split(".")
        if("encoder" in decomposed_key):
            pretrained_key = ".".join(decomposed_key[:])
            processed_dict[k] = model_compression_dict[pretrained_key] 
        if("decoder" in decomposed_key):
            pretrained_key = ".".join(decomposed_key[:])
            processed_dict[k] = model_compression_dict[pretrained_key]
        if("entropy_bottleneck" in decomposed_key):
            pretrained_key = ".".join(decomposed_key[:])
            processed_dict[k] = model_compression_dict[pretrained_key]
        # if("classifier" in decomposed_key):
        #     pretrained_key = ".".join(decomposed_key[1:])
        #     processed_dict[k] = model_classification_dict[pretrained_key] 
        for k in processed_dict.keys(): 
            model_dict[k] = processed_dict[k]
    # If load processed dict the model did not have the same performance
    # with pretrained model. I assume that the weight state dict dose not
    # as the same with the original. Therefore, I create a for loop to 
    # copy the weights to the original dict of the model.
    
    # check state dict 
    # models_differ = 0
    # for key_item_1, key_item_2 in zip(model_dict.items(), processed_dict.items()):
    #     if torch.equal(key_item_1[1], key_item_2[1]):
    #         pass
    #     else:
    #         models_differ += 1
    #         if (key_item_1[0] == key_item_2[0]):
    #             print('Mismtach found at', key_item_1[0])
    #         else:
    #             raise Exception
    # if models_differ == 0:
    #     print('Models match perfectly! :)')

    # model.load_state_dict(model_dict)
    
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    # filedirs = get_file_dirs(args.data_split_path)
    
    train_dataset = ModelNetH5_voxelize_all(data_root=args.data_split_path, 
                                            phase='train', 
                                            resolution=args.resolution, 
                                            num_points=1024)
    train_dataloader = make_data_loader_minkowski(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False, num_workers=4)

    # val_dataset = PCDataset(filedirs['Val'])
    # val_dataloader = make_data_loader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)
    
    test_dataset = ModelNetH5_voxelize_all(data_root=args.data_split_path, 
                                            phase='test', 
                                            resolution=args.resolution, 
                                            num_points=1024)
    test_dataloader = make_data_loader_minkowski(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)

        

    # accuracy = test(model, test_dataloader, 'cuda')
    # print(f"Test accuracy: {accuracy}")

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
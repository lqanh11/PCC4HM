import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
# from data_loader import PCDataset, make_data_loader
from data_loader_h5 import PCDataset_LoadAll, make_data_loader
from pcc_model_scalable import PCCModel, PCCModel_Scalable_ForBest
from classification_model import MinkowskiFCNN
from trainer_scalable_load_all import Trainer_Load_All
import random

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/')
    

    parser.add_argument("--alpha", type=float, default=1600., help="weights for distoration.")
    parser.add_argument("--gamma", type=float, default=0.06, help="weights for machine task.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--check_time", type=float, default=20,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='20231219_modelnet10_dense_FIXrec_TRAINbase_LEAVEres', help="prefix of checkpoints/logger, etc.")
 
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, gamma, lr, check_time):
        
        index = 0
        logdir_new = '{}_alpha_{}_{:03d}'.format(logdir, alpha, int(index))
        while os.path.exists(logdir_new):
            index+=1
            logdir_new = '{}_alpha_{}_{:03d}'.format(logdir, alpha, int(index))
            
        logdir = logdir_new    
        ckptdir = os.path.join(logdir, 'ckpts')
        
        print(logdir)
        
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lr = lr
        self.check_time=check_time

def get_file_dirs(root_path):

    train_file_dirs = []
    test_file_dirs = []

    ## ModelNet10
    for filename in os.listdir(os.path.join(root_path, 'train')):
        train_file_dirs.append(os.path.abspath(os.path.join(root_path, 'train', filename)))
    for filename in os.listdir(os.path.join(root_path, 'test')):
        test_file_dirs.append(os.path.abspath(os.path.join(root_path, 'test', filename)))

    random.shuffle(train_file_dirs)
    random.shuffle(test_file_dirs)

    print('Train: ', len(train_file_dirs), 
        'Test: ', len(test_file_dirs))
    
    return {'Train':train_file_dirs[:], 
            'Test':test_file_dirs[:]}


if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10', args.prefix), 
                            ckptdir='', 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta,
                            gamma=args.gamma,
                            lr=args.lr, 
                            check_time=args.check_time)
    
    

    # model
    model = PCCModel_Scalable_ForBest()
    model_dict = model.state_dict()


    state_dict = torch.load("/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/20231213_modelnet10_dense_FIXrec_TRAINbase_LEAVEres_alpha_16000.0_000/ckpts/epoch_10.pth")
    model_compression_dict = state_dict["model"]
    ## load pre-trained model
    # model_compression = PCCModel()

    # ckpt_compression = torch.load("/media/avitech/Data/quocanhle/PointCloud/compression_frameworks/PCGC/PCGCv2/logs_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth")
    # model_compression.load_state_dict(ckpt_compression['model'])
    # model_compression_dict = model_compression.state_dict()

    # model_classification = MinkowskiFCNN(in_channel=3, out_channel=10, embedding_channel=1024)
    # ckpt_cls = torch.load("/media/avitech/Data/quocanhle/PointCloud/logs/Mink_classification/classification_modelnet10_voxelize/2048/minkfcnn_voxelized_128/modelnet_minkfcnn.pth")
    # model_classification.load_state_dict(ckpt_cls['state_dict'])
    # model_classification_dict = model_classification.state_dict()

    # processed_dict = {}
    # for k in model_dict.keys(): 
    #     decomposed_key = k.split(".")
    #     if("encoder" in decomposed_key):
    #         pretrained_key = ".".join(decomposed_key[:])
    #         processed_dict[k] = model_compression_dict[pretrained_key] 
    #     if("decoder" in decomposed_key):
    #         pretrained_key = ".".join(decomposed_key[:])
    #         processed_dict[k] = model_compression_dict[pretrained_key]
    #     if("entropy_bottleneck" in decomposed_key):
    #         pretrained_key = ".".join(decomposed_key[:])
    #         processed_dict[k] = model_compression_dict[pretrained_key]
    #     # if("classifier" in decomposed_key):
    #     #     pretrained_key = ".".join(decomposed_key[1:])
    #     #     processed_dict[k] = model_classification_dict[pretrained_key] 

    #     for k in processed_dict.keys(): 
    #         model_dict[k] = processed_dict[k]
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
        # if("entropy_bottleneck_b" in decomposed_key):
        #     pretrained_key = ".".join(decomposed_key[:])
        #     processed_dict[k] = model_compression_dict[pretrained_key]
        # if("adapter" in decomposed_key):
        #     pretrained_key = ".".join(decomposed_key[:])
        #     processed_dict[k] = model_compression_dict[pretrained_key]
        # if("latentspace_transform" in decomposed_key):
        #     pretrained_key = ".".join(decomposed_key[:])
        #     processed_dict[k] = model_compression_dict[pretrained_key]
        # if("classifier" in decomposed_key):
        #     pretrained_key = ".".join(decomposed_key[:])
        #     processed_dict[k] = model_compression_dict[pretrained_key]

        for k in processed_dict.keys(): 
            model_dict[k] = processed_dict[k]

        model.load_state_dict(model_dict)
    

    # trainer    
    trainer = Trainer_Load_All(config=training_config, model=model)
    if args.init_ckpt != '':
        print(f'Load CKPT from {args.init_ckpt}')
        trainer.load_state_dict()

    # dataset
    filedirs = get_file_dirs(args.root_path)
    
    train_dataset = PCDataset_LoadAll(filedirs['Train'], resolution=128, num_pts=2048)
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False, num_workers=4)

    # val_dataset = PCDataset(filedirs['Val'])
    # val_dataloader = make_data_loader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)
    
    test_dataset = PCDataset_LoadAll(filedirs['Test'], resolution=128, num_pts=2048)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
        trainer.train(train_dataloader)
        # trainer.validation(val_dataloader, 'Validation')
        trainer.test(test_dataloader, 'Test')
    
import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
from data_loader_classification import PCDataset_Classification, make_data_loader
from pcc_model_scalable import PCCModel
from trainer_frozen import Trainer

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_split_path", default='/media/avitech/Data/quocanhle/PointCloud/data_split/modelnet10/resample/128_dense')
    

    parser.add_argument("--alpha", type=float, default=10., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--init_ckpt", default='/media/avitech/Data/quocanhle/PointCloud/compression_frameworks/PCGC/PCGCv2/logs_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='20231209_shapenet_reconstruction_frozen_000', help="prefix of checkpoints/logger, etc.")
 
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, lr, check_time):
        
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
        self.lr = lr
        self.check_time=check_time


def get_file_dirs(data_split_path):
    ## ModelNet10
    train_list_file = ['ply_data_train_file.csv']
    test_list_file = ['ply_data_test_file.csv']

    train_file_dirs = []
    test_file_dirs = []

    for train_file in train_list_file:
        df = pd.read_csv(os.path.join(data_split_path, train_file))
        for file_dir in df.values:
            train_file_dirs.append(file_dir[1])

    for test_file in test_list_file:
        df = pd.read_csv(os.path.join(data_split_path, test_file))
        for file_dir in df.values:
            test_file_dirs.append(file_dir[1])

    print('Train: ', len(train_file_dirs), 
        'Test: ', len(test_file_dirs))
    
    return {'Train':train_file_dirs[:], 
            'Test':test_file_dirs[:]}

if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/', args.prefix), 
                            ckptdir='', 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            lr=args.lr, 
                            check_time=args.check_time)    
    # model
    model = PCCModel()
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    
    
    # dataset
    filedirs = get_file_dirs(args.data_split_path)
    
    train_dataset = PCDataset_Classification(filedirs['Train'])
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False, num_workers=4)

    # val_dataset = PCDataset(filedirs['Val'])
    # val_dataloader = make_data_loader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)
    
    test_dataset = PCDataset_Classification(filedirs['Test'])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
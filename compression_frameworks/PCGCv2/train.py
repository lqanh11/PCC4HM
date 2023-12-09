import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
from data_loader import PCDataset, make_data_loader
from pcc_model import PCCModel
from trainer import Trainer

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--alpha", type=float, default=1., help="weights for distoration.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epoch", type=int, default=50)
    parser.add_argument("--check_time", type=float, default=10,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='20231209_shapenet_reconstruction_000', help="prefix of checkpoints/logger, etc.")
 
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
    # ShapeNet dataset
    dataset_path = '/media/avitech/Data/quocanhle/PointCloud/dataset/training_dataset/PCGCv2/'
    dataset_num = 2e4
    
    # model
    model = PCCModel()
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    
    
    filedirs = sorted(glob.glob(dataset_path+'*.h5'))[:int(dataset_num)]

    train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
    test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)

    # training
    for epoch in range(0, args.epoch):
        if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
        trainer.train(train_dataloader)
        trainer.test(test_dataloader, 'Test')
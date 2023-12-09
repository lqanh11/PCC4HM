import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import h5py
import random
random.seed()
from data_loader import PCDataset, make_data_loader
from trainer import Trainer
from pcc_model import PCCModel

def parse_args(): 
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, default='/media/avitech/Data/quocanhle/PointCloud/dataset/shapenet/points64/')
    parser.add_argument("--dataset_num", type=int, default=2.8e5) 
    parser.add_argument(
        "--alpha", type=float, default=10, dest="alpha", #6
        help="weights for distoration.")
    parser.add_argument(
        "--beta", type=float, default=1., dest="beta",
        help="Weight for empty position.")
    parser.add_argument(
        "--gamma", type=float, default=1, dest="gamma",
        help="Weight for hyper likelihoods.")
    parser.add_argument(
        "--delta", type=float, default=1, dest="delta",
        help="Weight for latent likelihoods.")
    parser.add_argument(
        "--lr", type=float, default=2e-4, dest="lr", #2e-4
        help="learning rate.")
    parser.add_argument("--epoch", type=int, default=30) 

    parser.add_argument(
        "--prefix", type=str, default='20231209', dest="prefix",
        help="prefix of checkpoints/logger.")
    parser.add_argument(
    "--init_ckpt", type=str, default='', dest="init_ckpt", 
    help='initial checkpoint directory.')

    parser.add_argument(
        "--lower_bound", type=float, default=1e-9, dest="lower_bound",
        help="lower bound of scale. 1e-5 or 1e-9")
    parser.add_argument(
    "--batch_size", type=int, default=8, dest="batch_size", 
    help='batch_size')
 
    args = parser.parse_args()

    return args

class TrainingConfig(): 
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, beta, gamma, delta, lr):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.beta = beta
        self.lr = lr
        self.gamma = gamma # weight of hyper prior.
        self.delta = delta # weight of latent representation.

#已改
if __name__ == '__main__':
    # log
    args = parse_args()
    # Define parameters.
    RATIO_EVAL = 9 


    training_config = TrainingConfig(
                            logdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv1/train/', args.prefix), 
                            ckptdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/PCGCv1/train/', args.prefix),
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta, 
                            gamma=args.gamma,
                            delta=args.delta,
                            lr=args.lr)
    # model
    model = PCCModel(lower_bound=args.lower_bound)
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    filedirs = sorted(glob.glob(args.dataset+'*.h5'))[:int(args.dataset_num)]

    # training
    for epoch in range(0, args.epoch):
        if epoch>0:
            trainer.update_lr(lr=max(trainer.config.lr/2, 1e-5)) #update lr
        train_list = random.sample(filedirs[len(filedirs)//RATIO_EVAL:], 100*args.batch_size) 
        train_dataset = PCDataset(train_list)
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, repeat=False)
        trainer.train(train_dataloader)
        
        eval_list = random.sample(filedirs[:len(filedirs)//RATIO_EVAL], 20*args.batch_size) #10
        test_dataset = PCDataset(eval_list)
        test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3, repeat=False)
        trainer.test(test_dataloader, 'Test')

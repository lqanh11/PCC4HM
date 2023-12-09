#不要改学习率，基于源码学习率，调整scale_lower_bound，继续训练
import time, os, sys, glob, argparse
import importlib
import numpy as np
import torch
import random
random.seed(3)
from data_loader import PCDataset, make_data_loader
from trainer import Trainer
from model_configs import C3PCompressionModelV2

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, default='/userhome/pcc_geo_cnn_v2/pcc_geo_cnn_v2-master/ModelNet40_200_pc512_oct3_4k/')
    parser.add_argument(
        "--alpha", type=float, default=0.75, dest="alpha",
        help='Focal loss alpha.')
    parser.add_argument(
        "--lmbda", type=float, default=3.0e-4, dest="lmbda",
        help='Lambda for rate-distortion tradeoff.')
    parser.add_argument(
        "--gamma", type=float, default=2., dest="gamma",
        help='Focal loss gamma.')
    parser.add_argument(
        '--num_filters', type=int, default=64,
        help='Number of filters per layer.')
    parser.add_argument(
        '--resolution',
        type=int, help='Dataset resolution.', default=64)
    parser.add_argument(
        '--batch_size', type=int, default=48,
        help='Batch size for training.')
    # parser.add_argument(
    #     '--max_steps', type=int, default=100000,
    #     help='Train up to this number of steps.')
    parser.add_argument(
        "--main_lr", type=float, default=1e-4, dest="main_lr", #4e-4
        help="learning rate.")
    parser.add_argument(
        "--aux_lr", type=float, default=1e-3, dest="aux_lr", #4e-4
        help="learning rate.")
    parser.add_argument("--epoch", type=int, default=50) #
    parser.add_argument(
        "--save_dir", type=str, default='/userhome/pcc_geo_cnn_v2/pytorch/models/', dest="save_dir",
        help='save checkpoint and log directory.')
    parser.add_argument(
        "--init_ckpt", type=str, default='/userhome/pcc_geo_cnn_v2/pytorch/models/01/epoch_214_15253.pth', dest="init_ckpt",
        help='initial checkpoint directory.')
    parser.add_argument(
        "--prefix", type=str, default='02', dest="prefix",
        help="prefix of checkpoints/logger.")
    parser.add_argument(
        "--scale_lower_bound", type=float, default=1e-9, dest="scale_lower_bound",
        help="lower bound of scale. 1e-5 or 1e-9")
 
    args = parser.parse_args()

    return args

class TrainingConfig(): #已改
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, lmbda, gamma, resolution, main_lr,aux_lr):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.lmbda = lmbda
        self.gamma = gamma # weight of hyper prior.
        self.main_lr = main_lr
        self.aux_lr = aux_lr
        self.resolution = resolution

#已改
if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join(args.save_dir, args.prefix), 
                            ckptdir=os.path.join(args.save_dir, args.prefix), #保存当前训练的模型
                            init_ckpt=args.init_ckpt,  #初始化模型
                            alpha=args.alpha, 
                            lmbda=args.lmbda, 
                            gamma=args.gamma,
                            resolution=args.resolution,
                            main_lr=args.main_lr,
                            aux_lr=args.aux_lr
                            )
    # model
    model = C3PCompressionModelV2(num_filters=args.num_filters,scale_lower_bound=args.scale_lower_bound)
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    dataset = args.dataset + "**/*.ply"
    filedirs = np.array(sorted(glob.glob(dataset, recursive=True)))
    print("all files len(filedirs):",len(filedirs)) #all files len(filedirs): 4000
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in filedirs])
    for cat in files_cat:
        assert (cat == 'train') or (cat == 'test')
    files_train = filedirs[files_cat == 'train']
    files_test = filedirs[files_cat == 'test']
    assert(len(files_train) + len(files_test) == len(filedirs))
    print("len(files_train):",len(files_train)) #len(files_train): 3379
    print("len(files_test):",len(files_test)) #len(files_test): 621
    # training
    for epoch in range(0, args.epoch):
        if (epoch+1) %3 == 0: 
            ori_lr = trainer.config.main_lr
            trainer.config.main_lr =  max(trainer.config.main_lr/2, 1e-4)# update lr
            if trainer.config.main_lr < ori_lr:
                trainer.set_optimizer()
            
            ori_lr = trainer.config.aux_lr
            trainer.config.aux_lr =  max(trainer.config.aux_lr/2, 1e-3)# update lr
            if trainer.config.aux_lr < ori_lr:
                trainer.set_optimizer()
        train_dataset = PCDataset(files_train)
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=6, repeat=False)
        trainer.train(train_dataloader)

        test_dataset = PCDataset(files_test)
        test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3, repeat=False)
        trainer.test(test_dataloader)

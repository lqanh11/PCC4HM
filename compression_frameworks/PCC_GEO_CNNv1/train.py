import os, glob, argparse
# import torch
import random
random.seed(3)
from data_loader import PCDataset, make_data_loader
from trainer import Trainer
from compression_model import PCCModel
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--dataset", type=str,
        default='/media/avitech/Data/quocanhle/PointCloud/dataset/training_dataset/PCC_GEO_CNNv1/ModelNet40_pc_64') 
    parser.add_argument(
        '--lmbda', type=float, default=5e-5, # 0.0001
        help='Lambda for rate-distortion tradeoff.')
    parser.add_argument(
        '--alpha', type=float, default=0.9,
        help='Focal loss alpha.')
    parser.add_argument(
        '--gamma', type=float, default=2.0,
        help='Focal loss gamma.')
    parser.add_argument(
        "--lr", type=float, default=1e-4, dest="lr", #4e-4
        help="learning rate.")
    parser.add_argument("--epoch", type=int, default=20) #
    parser.add_argument(
    "--init_ckpt", type=str, default='/media/avitech/Data/quocanhle/PointCloud/pretrained_models/PCC_GEO_CNNv1/models/0.001/epoch_24_30503.pth', dest="init_ckpt",
    help='initial checkpoint directory.')
    parser.add_argument(
    "--batch_size", type=int, default=8, dest="batch_size",
    help='batch_size')
    parser.add_argument(
        '--resolution', type=int, help='Dataset resolution.', default=64)
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')

    args = parser.parse_args()

    return args

class TrainingConfig(): 
    def __init__(self, logdir, ckptdir, init_ckpt, alpha, lmbda, gamma, lr,aux_lr,resolution):
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = init_ckpt
        self.alpha = alpha
        self.lmbda = lmbda
        self.lr = lr
        self.aux_lr = aux_lr
        self.resolution = resolution
        self.gamma = gamma # weight of hyper prior.

if __name__ == '__main__':
    args = parse_args()
    training_config = TrainingConfig(
                            logdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/PCC_GEO_CNNv1', 'models/'+str(args.lmbda)), 
                            ckptdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/PCC_GEO_CNNv1','models/'+str(args.lmbda)+'/ckpts'), 
                            init_ckpt=args.init_ckpt,  
                            alpha=args.alpha, 
                            lmbda=args.lmbda, 
                            gamma=args.gamma,
                            lr=args.lr,
                            aux_lr=1e-3,
                            resolution=args.resolution
                            )
    # model
    model = PCCModel(num_filters=args.num_filters)
    # trainer    
    trainer = Trainer(config=training_config, model=model)

    # dataset
    dataset = args.dataset + "/*/*/*.ply"
    # print(dataset)
    filedirs = np.array(sorted(glob.glob(dataset, recursive=True)))
    # print(filedirs)
    print("all files len(filedirs):",len(filedirs))
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in filedirs])
    for cat in files_cat:
        assert (cat == 'train') or (cat == 'test')
    files_train = filedirs[files_cat == 'train']
    files_test = filedirs[files_cat == 'test']
    assert(len(files_train) + len(files_test) == len(filedirs))
    print("len(files_train):",len(files_train)) #7589
    print("len(files_test):",len(files_test)) #1860
    # training
    for epoch in range(0, args.epoch):
        if epoch %3 == 2: 
            ori_lr = trainer.config.lr
            trainer.config.lr =  max(trainer.config.lr/2, 1e-4)# update lr
            if trainer.config.lr < ori_lr:
                trainer.main_optimizer,_ = trainer.set_optimizer()
            
            ori_lr = trainer.config.aux_lr
            trainer.config.aux_lr =  max(trainer.config.aux_lr/2, 1e-3)# update lr
            if trainer.config.aux_lr < ori_lr:
                _,trainer.aux_optimizer = trainer.set_optimizer()
        train_dataset = PCDataset(files_train)
        train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, repeat=False)
        trainer.train(train_dataloader)

        test_dataset = PCDataset(files_test)
        test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=3, repeat=False)
        trainer.test(test_dataloader)

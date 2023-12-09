import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
# from data_loader import PCDataset, make_data_loader
from data_loader_classification import PCDataset_Classification, make_data_loader
from pcc_model_scalable import PCCModel_Scalable, PCCModel

from classification_model import MinkowskiPointNet
from trainer_scalable import Trainer_Scalable

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--data_split_path", default='/media/avitech/Data/quocanhle/PointCloud/data_split/modelnet10/resample/128_dense')
    

    parser.add_argument("--alpha", type=float, default=10., help="weights for distoration.")
    parser.add_argument("--gamma", type=float, default=0.06, help="weights for machine task.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--init_ckpt", default='')
    parser.add_argument("--lr", type=float, default=8e-4)

    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--check_time", type=float, default=20,  help='frequency for recording state (min).') 
    parser.add_argument("--prefix", type=str, default='scalable_modelnet_dense_full_reconstruction_with_pretrained_cls_1024_train_together', help="prefix of checkpoints/logger, etc.")
 
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

def get_file_dirs(data_split_path):
    ## ModelNet40
    # train_list_file = ['ply_data_train_0_id2file.csv',
    #                    'ply_data_train_1_id2file.csv',
    #                    'ply_data_train_2_id2file.csv',
    #                    'ply_data_train_3_id2file.csv']

    # val_list_file = ['ply_data_train_4_id2file.csv']

    # test_list_file = ['ply_data_test_0_id2file.csv',
    #                   'ply_data_test_1_id2file.csv']

    # train_file_dirs = []
    # val_file_dirs = []
    # test_file_dirs = []

    # for train_file in train_list_file:
    #     df = pd.read_csv(os.path.join(data_split_path, train_file))
    #     for file_dir in df.values:
    #         train_file_dirs.append(file_dir[1])

    # for val_file in val_list_file:
    #     df = pd.read_csv(os.path.join(data_split_path, val_file))
    #     for file_dir in df.values:
    #         val_file_dirs.append(file_dir[1])

    # for test_file in test_list_file:
    #     df = pd.read_csv(os.path.join(data_split_path, test_file))
    #     for file_dir in df.values:
    #         test_file_dirs.append(file_dir[1])

    # print('Train: ', len(train_file_dirs), 
    #     'Val: ', len(val_file_dirs),
    #     'Test: ', len(test_file_dirs))
    
    # return {'Train':train_file_dirs, 
    #         'Val':val_file_dirs, 
    #         'Test':test_file_dirs}

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
                            logdir=os.path.join('/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10', args.prefix), 
                            ckptdir='', 
                            init_ckpt=args.init_ckpt, 
                            alpha=args.alpha, 
                            beta=args.beta,
                            gamma=args.gamma,
                            lr=args.lr, 
                            check_time=args.check_time)
    
    

    # model
    model = PCCModel_Scalable()
    model_dict = model.state_dict()

    ## load pre-trained model
    model_compression = PCCModel()

    ckpt_compression = torch.load("/media/avitech/Data/quocanhle/PointCloud/compression_frameworks/PCGC/PCGCv2/logs_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth")
    model_compression.load_state_dict(ckpt_compression['model'])
    model_compression_dict = model_compression.state_dict()

    model_classification = MinkowskiPointNet(in_channel=3, out_channel=10, embedding_channel=1024)
    ckpt_cls = torch.load("/media/avitech/Data/quocanhle/PointCloud/analysis_frameworks/Minkowski_CNN_cls/src/cls_logs/classification_modelnet10_voxelize/1024/minkpointnet/modelnet_minkpointnet.pth")
    model_classification.load_state_dict(ckpt_cls['state_dict'])
    model_classification_dict = model_classification.state_dict()

    with torch.no_grad():
        processed_dict = {}

        for k in model_dict.keys(): 
            decomposed_key = k.split(".")
            if("encoder" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict[pretrained_key] 
            if("decode" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict[pretrained_key]
            if("entropy_bottleneck" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict[pretrained_key] 
            if("classifier" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[1:])
                processed_dict[k] = model_classification_dict[pretrained_key] 

        model.load_state_dict(processed_dict, strict=False)
    

    # trainer    
    trainer = Trainer_Scalable(config=training_config, model=model)
    if args.init_ckpt != '':
        print(f'Load CKPT from {args.init_ckpt}')
        trainer.load_state_dict()

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
        # trainer.validation(val_dataloader, 'Validation')
        trainer.test(test_dataloader, 'Test')
        
# if __name__ == '__main__':
#     # log
#     args = parse_args()
#     training_config = TrainingConfig(
#                             logdir=os.path.join('../logs', args.prefix), 
#                             ckptdir='', 
#                             init_ckpt=args.init_ckpt, 
#                             alpha=args.alpha, 
#                             beta=args.beta, 
#                             lr=args.lr, 
#                             check_time=args.check_time)
#     # ShapeNet dataset
#     dataset_path = '/media/avitech/Data/quocanhle/PointCloud/dataset/training_dataset/'
#     dataset_num = 2e4
    
#     # model
#     model = PCCModel()
#     # trainer    
#     trainer = Trainer(config=training_config, model=model)

#     # dataset
    
    
#     filedirs = sorted(glob.glob(dataset_path+'*.h5'))[:int(dataset_num)]

#     train_dataset = PCDataset(filedirs[round(len(filedirs)/10):])
#     train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False)
#     test_dataset = PCDataset(filedirs[:round(len(filedirs)/10)])
#     test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)

#     # training
#     for epoch in range(0, args.epoch):
#         if epoch>0: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
#         trainer.train(train_dataloader)
#         trainer.test(test_dataloader, 'Test')
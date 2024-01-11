import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME

from data_loader_h5 import PCDataset_LoadAll, make_data_loader
from pcc_model_scalable import PCCModel, PCCModel_Classification_MLP_Scalable_Base
from scalable_MLP_base_trainer import Trainer
import random
import shutil
import datetime

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/')
    
    parser.add_argument("--alpha", type=float, default=.5, help="weights for distoration.")
    parser.add_argument("--gamma", type=float, default=0.06, help="weights for machine task.")
    parser.add_argument("--beta", type=float, default=1., help="weights for bit rate.")

    parser.add_argument("--logdir", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only')
    
    parser.add_argument("--init_ckpt_original", default='/media/avitech/Data/quocanhle/PointCloud/compression_frameworks/PCGC/PCGCv2/logs_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth')
    parser.add_argument("--init_ckpt_base", default='')
    parser.add_argument("--init_ckpt", default='')

    parser.add_argument('--params_to_train', 
                        type=str, 
                        nargs='*',
                        default=[
                                #    'encoder', # original 
                                #    'decoder', # original
                                #    'entropy_bottleneck', # original
                                    'entropy_bottleneck_b', # base
                                    'classifier_backbone', # base
                                    'classifier_mlp',
                                    'reconstruction_gain', # base
                                #    'entropy_bottleneck_e', # enhancemet
                                   'reconstruction_backbone', # enhancemet
                                #    'analysis_residual', # enhancemet
                                #    'systhesis_residual' # enhancemet
                                            ])

    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--check_time", type=float, default=20,  help='frequency for recording state (min).') 
    parser.add_argument("--resolution", type=int, default=128, help="resolution")
    parser.add_argument("--prefix", type=str, default='encFIXa10_baseTRANc4_MLP_scalable', help="prefix of checkpoints/logger, etc.")
 
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, args):

        timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
        
        index = 0
        logdir_new = '{}_resolution{}_alpha{}_{:03d}'.format(os.path.join(args.logdir, timestr + '_' + args.prefix), args.resolution, args.alpha, int(index))
        while os.path.exists(logdir_new):
            index+=1
            logdir_new = '{}_resolution{}_alpha{}_{:03d}'.format(os.path.join(args.logdir, timestr + '_' + args.prefix), args.resolution, args.alpha, int(index))
            
        logdir = logdir_new    
        ckptdir = os.path.join(logdir, 'ckpts')
        
        print(logdir)
        
        src_dir = os.path.join(logdir, 'src')
        os.makedirs(src_dir, exist_ok=True)
        ## copy file to log dir
        shutil.copy('data_loader_h5.py' , str(src_dir))
        shutil.copy('pcc_model_scalable.py', str(src_dir))
        shutil.copy('scalable_MLP_base_train.py', str(src_dir))
        shutil.copy('scalable_MLP_base_trainer.py', str(src_dir))
        shutil.copy('loss.py', str(src_dir))
        shutil.copy('data_utils.py', str(src_dir))
        shutil.copy('autoencoder.py', str(src_dir))
        shutil.copy('entropy_model.py', str(src_dir))
        shutil.copy('classification_model.py', str(src_dir))
        
        if args.init_ckpt_original != '':
            save_ckpt_original = os.path.join(src_dir, 'ckpt_original')
            os.makedirs(save_ckpt_original, exist_ok=True)
            shutil.copy(args.init_ckpt_original, str(save_ckpt_original))
            self.init_ckpt_original = os.path.join(save_ckpt_original, os.path.split(args.init_ckpt_original)[-1])
        else:
            self.init_ckpt_original = ''
        
        if args.init_ckpt_base != '':
            save_ckpt_base = os.path.join(src_dir, 'ckpt_base')
            os.makedirs(save_ckpt_base, exist_ok=True)     
            shutil.copy(args.init_ckpt_base, str(save_ckpt_base))
            self.init_ckpt_base = os.path.join(save_ckpt_base, os.path.split(args.init_ckpt_base)[-1])
        else:
            self.init_ckpt_base = ''
        
        self.logdir = logdir
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)
        self.ckptdir = ckptdir
        if not os.path.exists(self.ckptdir): os.makedirs(self.ckptdir)
        self.init_ckpt = args.init_ckpt
        self.alpha = args.alpha
        self.beta = args.beta
        self.gamma = args.gamma
        self.resolution = args.resolution
        self.lr = args.lr
        self.check_time = args.check_time
        self.save_best_model_path = os.path.join(logdir, 'best_model')

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
    training_config = TrainingConfig(args)
    
    # model
    model = PCCModel_Classification_MLP_Scalable_Base()
    model_dict = model.state_dict() 
    processed_dict = {}

    # load pre-trained model
    if training_config.init_ckpt_original != '':
        state_dict = torch.load(training_config.init_ckpt_original)
        model_compression_dict_original = state_dict["model"]
        
        for k in model_dict.keys(): 
            decomposed_key = k.split(".")
            if("encoder" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_original[pretrained_key] 
            # if("decoder" in decomposed_key):
            #     pretrained_key = ".".join(decomposed_key[:])
            #     processed_dict[k] = model_compression_dict_original[pretrained_key]
            if("entropy_bottleneck" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_original[pretrained_key]
    # load base branch
    if training_config.init_ckpt_base != '':
        state_dict = torch.load(training_config.init_ckpt_base)
        model_compression_dict_base = state_dict["model"]
        for k in model_dict.keys(): 
            decomposed_key = k.split(".")
            if("entropy_bottleneck_b" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("adapter" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("latentspacetransform" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("classifier" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]

    # load enhanment brach
    if training_config.init_ckpt != '':
        state_dict = torch.load(training_config.init_ckpt)
        model_compression_dict_enhance = state_dict["model"]
        for k in model_dict.keys(): 
            decomposed_key = k.split(".")

            if("entropy_bottleneck_e" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("transpose_adapter" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("analysis_residual" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("systhesis_residual" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
    
    for k in processed_dict.keys(): 
        # print('Load weight: ', k)
        model_dict[k] = processed_dict[k]

    model.load_state_dict(model_dict)
    

    # trainer    
    trainer = Trainer(config=training_config, model=model)
    if args.init_ckpt != '':
        print(f'Load CKPT from {args.init_ckpt}')
        trainer.load_state_dict()

    # dataset
    filedirs = get_file_dirs(args.root_path)
    
    train_dataset = PCDataset_LoadAll(filedirs['Train'], resolution=args.resolution, num_pts=1024)
    train_dataloader = make_data_loader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, repeat=False, num_workers=4)

    # val_dataset = PCDataset(filedirs['Val'])
    # val_dataloader = make_data_loader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)
    
    test_dataset = PCDataset_LoadAll(filedirs['Test'], resolution=args.resolution, num_pts=1024)
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False, num_workers=4)

    
    # Frezee layers in model
    # all
    params_to_train = args.params_to_train
    # training
    # Set up early stopping parameters
    patience = 5  # Number of epochs with no improvement after which training will be stopped
    best_val_loss = float('inf')
    current_patience = 0

    for epoch in range(0, args.epoch):
        if epoch>5: trainer.config.lr =  max(trainer.config.lr/2, 1e-5)# update lr 
        model_path = trainer.train(train_dataloader, params_to_train)
        # trainer.validation(val_dataloader, 'Validation')
        val_loss = trainer.test(test_dataloader, 'Test')

        if epoch > 30:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = model_path
                current_patience = 0
            else:
                current_patience += 1
            
            if current_patience >= patience:
                print(f"Early stopping after {epoch + 1} epochs.")
                os.makedirs(training_config.save_best_model_path, exist_ok=True)
                shutil.copy(best_model_path , str(training_config.save_best_model_path))
                break
            elif epoch == (args.epoch - 1):
                best_model_path = model_path
                os.makedirs(training_config.save_best_model_path, exist_ok=True)
                shutil.copy(best_model_path , str(training_config.save_best_model_path))


    
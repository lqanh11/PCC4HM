import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME

from data_loader_h5 import ModelNet_latentspace, make_data_loader_latentspace
from pcc_model_scalable import PCCModel_Classification_MLP_Scalable_Full
from tester_scalable_two_tasks_MLP import Test_Load_All
import random
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/latentspace/finetuning')

    
    parser.add_argument("--outdir",
                        default="/media/avitech/Data/quocanhle/PointCloud/source_code/scalable_frameworks/PCGC/evaluation/output/ModelNet/scalable_two_tasks")
    
    parser.add_argument("--logdir", 
                        default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/')

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1)
    
    parser.add_argument("--resolution", 
                        type=int, 
                        default=128, 
                        help="resolution")
    
    parser.add_argument("--rate", 
                        type=str, 
                        default='r7', 
                        help="compression rate")

    parser.add_argument("--prefix", 
                        type=str, 
                        default='2024-01-10_17-56_encFIXa10_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000', 
                        help="prefix of checkpoints/logger, etc.")
 
    
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, args):

        self.logdir = os.path.join(args.logdir, args.prefix)
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)        
        
        ckpt_original_path = os.path.join(self.logdir, 'src','ckpt_original')
        if os.path.exists(ckpt_original_path):
            files = os.listdir(ckpt_original_path)
            self.init_ckpt_original = os.path.join(ckpt_original_path, files[0])
        else:
            self.init_ckpt_original = ''
        
        ckpt_base_path = os.path.join(self.logdir, 'src', 'ckpt_base')
        if os.path.exists(ckpt_base_path):
            files = os.listdir(ckpt_base_path)
            self.init_ckpt_base = os.path.join(ckpt_base_path, files[0])
        else:
            self.init_ckpt_base = ''
        
        ckpt_enhanment_path = os.path.join(self.logdir, 'best_model')
        if os.path.exists(ckpt_enhanment_path):
            files = os.listdir(ckpt_enhanment_path)
            self.init_ckpt = os.path.join(ckpt_enhanment_path, files[0])
        else:
            self.init_ckpt = ''
       
        self.resolution = args.resolution
        self.outdir = args.outdir
        self.rate = args.rate
        

if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(args)
    
    # model
    model = PCCModel_Classification_MLP_Scalable_Full()
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
            if("decoder" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_original[pretrained_key]
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
            if("classifier_backbone" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("classifier_mlp" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("reconstruction_gain" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("reconstruction_backbone" in decomposed_key):
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
            if("reconstruction_gain" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("reconstruction_backbone" in decomposed_key):
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
    model.eval()
    
    # trainer    
    trainer = Test_Load_All(config=training_config, model=model)
    
    test_dataset = ModelNet_latentspace(data_root=args.root_path, 
                                        entropy_model=model.entropy_bottleneck,
                                        phase='test',
                                        resolution=args.resolution, 
                                        rate=args.rate)
    test_dataloader = make_data_loader_latentspace(dataset=test_dataset, 
                                                   batch_size=args.batch_size, 
                                                   shuffle=False, 
                                                   repeat=False, 
                                                   num_workers=4)
    

    trainer.test(test_dataloader, 'Test')

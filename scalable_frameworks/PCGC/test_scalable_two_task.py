import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME

from data_loader_h5 import ModelNet_latentspace, make_data_loader_latentspace
from pcc_model_scalable import PCCModel_Scalable_ForBest_KeepCoords
from tester_scalable_two_tasks import Test_Load_All
import random
import shutil

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/latentspace/finetuning')

    parser.add_argument("--logdir", 
                        default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/')

    parser.add_argument("--batch_size", 
                        type=int, 
                        default=1)
    
    parser.add_argument("--resolution", 
                        type=int, 
                        default=256, 
                        help="resolution")
    
    parser.add_argument("--rate", 
                        type=str, 
                        default='r2', 
                        help="compression rate")

    parser.add_argument("--prefix", 
                        type=str, 
                        default='2024-01-06_04-27_encFIXa05_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution256_alpha1.0_000', 
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
        

if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(args)
    
    # model
    model = PCCModel_Scalable_ForBest_KeepCoords(
        encoder_channels = [1,16,32,64,32,8],
        decoder_channels = [8,64,32,16],
        adapter_channels = [8,4,4],
        t_adapter_channels = [4,4,8],

        num_cls = 10,
        embedding_channel_cls = 1024,

        analysis_channels = [8,32,8],
        synthesis_channels = [8,32,8],
        layer_blocks = 3
    )
    model_dict = model.state_dict() 
    processed_dict = {}

    # load pre-trained model
    if training_config.init_ckpt_original != '':
        print('Load original model from ',training_config.init_ckpt_original)
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
        print('Load base branch model from ',training_config.init_ckpt_base)
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
        print('Load enhanment branch model from ',training_config.init_ckpt)
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

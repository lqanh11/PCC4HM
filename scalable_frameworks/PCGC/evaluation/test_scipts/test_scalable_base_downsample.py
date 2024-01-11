import time, os, sys, glob, argparse
import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME

from data_loader_h5 import ModelNet_latentspace, make_data_loader_latentspace
from pcc_model_scalable import PCCModel_Classification_Adapter
from tester_scalable_base_downsample import Test_Load_All
import random
import shutil

# /media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/DownsampleCoordinates/20231230_encFIXa10_baseTRANc4_resolution128_alpha16000.0_000/ckpts/epoch_6.pth
# /media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/DownsampleCoordinates/20231230_encFIXa10_baseTRANc4_resolution128_alpha640.0_000/best_model/epoch_139.pth
# /media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/DownsampleCoordinates/20231230_encFIXa10_baseTRANc4_resolution128_alpha320.0_000/best_model/epoch_177.pth
# /media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/DownsampleCoordinates/20231230_encFIXa10_baseTRANc4_resolution128_alpha160.0_000/best_model/epoch_188.pth

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/latentspace/finetuning')

    parser.add_argument("--logdir", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/test_cls')

    parser.add_argument("--init_ckpt_original", default='/media/avitech/Data/quocanhle/PointCloud/compression_frameworks/PCGC/PCGCv2/logs_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth')
    parser.add_argument("--init_ckpt_base", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/cls_only/DownsampleCoordinates/20231230_encFIXa10_baseTRANc4_resolution128_alpha160.0_000/best_model/epoch_188.pth')
    parser.add_argument("--init_ckpt", default='')

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=128, help="resolution")
    parser.add_argument("--rate", type=str, default='r7', help="compression rate")

    parser.add_argument("--prefix", type=str, default='20240104_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords', help="prefix of checkpoints/logger, etc.")
 
    
    args = parser.parse_args()

    return args

class TrainingConfig():
    def __init__(self, args):
        
        
        if args.init_ckpt_original != '':
            self.init_ckpt_original = args.init_ckpt_original
        else:
            self.init_ckpt_original = ''
        
        if args.init_ckpt_base != '':
            self.init_ckpt_base = args.init_ckpt_base
        else:
            self.init_ckpt_base = ''
        
        self.init_ckpt = args.init_ckpt
        self.resolution = args.resolution
        self.logdir = os.path.join(args.logdir, args.prefix)
        if not os.path.exists(self.logdir): os.makedirs(self.logdir)

if __name__ == '__main__':
    # log
    args = parse_args()
    training_config = TrainingConfig(args)
    
    # model
    model = PCCModel_Classification_Adapter()
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
            if("adapter" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("classifier" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]

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

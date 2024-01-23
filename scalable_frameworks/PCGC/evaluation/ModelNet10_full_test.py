import time, os, sys, glob, argparse

sys.path.append('../')

import importlib
import numpy as np
import pandas as pd
import torch
import MinkowskiEngine as ME
from data_loader_h5_test import PCDataset_LoadAll, make_data_loader
from pcc_model_scalable import PCCModel_Classification_MLP_Scalable_Full
from ModelNet10_full_tester import Tester

def parse_args():
    parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--root_path", default='/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution')


    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=128)
    

    parser.add_argument("--logdir", type=str, default='/media/avitech/QuocAnh_1TB/Point_Cloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/OLD_used_in_the_thesis')
    parser.add_argument("--prefix", type=str, default='2024-01-13_12-39_encFIXa025_baseTRANc4_MLP_scalable_full_resolution128_alpha0.5_000', help="prefix of checkpoints/logger, etc.")
    parser.add_argument("--rate", type=str, default="r1")
    parser.add_argument("--outdir", type=str, default="./output/ModelNet/scalable_full/")
    

    args = parser.parse_args()

    return args

class TestingConfig():
    def __init__(self, args):
        
        self.logdir = os.path.join(args.logdir, args.prefix)

        filelist = os.listdir(os.path.join(self.logdir,  'src', 'ckpt_original'))
        self.init_ckpt_original = os.path.join(self.logdir,  'src', 'ckpt_original', filelist[0])
        filelist = os.listdir(os.path.join(self.logdir,  'src', 'ckpt_base'))
        self.init_ckpt_base = os.path.join(self.logdir, 'src','ckpt_base', filelist[0])
        filelist = os.listdir(os.path.join(self.logdir,  'best_model'))
        self.init_ckpt = os.path.join(self.logdir, 'best_model',filelist[0])

        
        self.resolution = args.resolution
        self.outdir = os.path.join(args.outdir, str(args.resolution), args.rate)
        os.makedirs(self.outdir, exist_ok=True)

def get_file_dirs(root_path):

    train_file_dirs = []
    test_file_dirs = []

    ## ModelNet10
    for filename in os.listdir(os.path.join(root_path, 'train')):
        train_file_dirs.append(os.path.abspath(os.path.join(root_path, 'train', filename)))
    for filename in os.listdir(os.path.join(root_path, 'test')):
        test_file_dirs.append(os.path.abspath(os.path.join(root_path, 'test', filename)))
    
    test_file_dirs = [
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0117.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0128.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0130.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0126.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0147.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0590.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0550.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0534.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0605.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0603.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0927.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0896.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0989.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0934.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0936.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0205.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0283.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0258.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0277.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0217.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0254.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0244.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0206.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0252.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0219.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0497.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0481.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0526.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0556.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0512.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0280.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0247.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0226.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0202.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0285.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0696.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0765.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0735.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0689.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0733.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0472.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0466.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0443.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0451.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0446.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0429.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0404.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0443.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0375.h5',
        '/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0360.h5',
    ]

    print('Train: ', len(train_file_dirs), 
        'Test: ', len(test_file_dirs))
    
    return {'Train':train_file_dirs[:], 
            'Test':test_file_dirs[:]}
        
if __name__ == '__main__':
    # log
    args = parse_args()
    testing_config = TestingConfig(args)
                            
    # model
    model = PCCModel_Classification_MLP_Scalable_Full()
    model_dict = model.state_dict() 
    processed_dict = {}

    # load pre-trained model
    if testing_config.init_ckpt_original != '':
        state_dict = torch.load(testing_config.init_ckpt_original)
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
    if testing_config.init_ckpt_base != '':
        state_dict = torch.load(testing_config.init_ckpt_base)
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
    # load enhanment brach
    if testing_config.init_ckpt != '':
        state_dict = torch.load(testing_config.init_ckpt)
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

    # trainer    
    tester = Tester(config=testing_config, model=model)

    # dataset
    filedirs = get_file_dirs(args.root_path)
    test_dataset = PCDataset_LoadAll(filedirs['Test'], 
                                    resolution=args.resolution, 
                                     num_pts=1024
                                    )
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False, repeat=False)

    tester.test(test_dataloader, 'Test')
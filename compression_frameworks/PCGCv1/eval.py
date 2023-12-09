#!/usr/bin/env python
# coding: utf-8

# Copyright (c) Nanjing University, Vision Lab.
# Last update: 
# 2021.9.11

import os
import time
import numpy as np
import matplotlib.pylab  as plt
import pandas as pd
import subprocess
import glob
import configparser
import argparse
import importlib 
import torch

from pcc_model import PCCModel
from process import preprocess, postprocess
from transform import compress_hyper, decompress_hyper
from dataprocess.inout_bitstream import write_binary_files_hyper, read_binary_files_hyper

os.environ['CUDA_VISIBLE_DEVICES']="0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from myutils.pc_error_wrapper import pc_error
from myutils.pc_error_wrapper import get_points_number

def test_hyper(input_file, resultdir_path, model, scale, cube_size, min_num, batch_parallel, postfix=''):
    # Pre-process
    cubes, cube_positions, points_numbers = preprocess(input_file, scale, cube_size, min_num)
    ### Encoding
    y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape =  compress_hyper(cubes, model, device, batch_parallel)
    # Write files
    filename = os.path.split(input_file)[-1][:-4]
    print(filename)
    resultdir = os.path.join(resultdir_path, './compressed'+ postfix +'/')
    bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos = write_binary_files_hyper(
        filename, y_strings, z_strings, points_numbers, cube_positions, 
        y_min_vs, y_max_vs, y_shape.numpy(), 
        z_min_v, z_max_v, z_shape.numpy(), resultdir)
    # Read files
    y_strings_d, z_strings_d, points_numbers_d, cube_positions_d,  y_min_vs_d, y_max_vs_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d =  \
        read_binary_files_hyper(filename, resultdir)
    # Decoding
    cubes_d = decompress_hyper(y_strings_d, y_min_vs_d, y_max_vs_d, 
                                y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model, device, batch_parallel)
    # bpp
    N = get_points_number(input_file)
    bpp = round(8*(bytes_strings + bytes_strings_head + bytes_strings_hyper + 
                    bytes_pointnums + bytes_cubepos)/float(N), 4)

    bpp_strings = round(8*bytes_strings/float(N), 4)
    bpp_strings_hyper = round(8*bytes_strings_hyper/float(N), 4)
    bpp_strings_head = round(8*bytes_strings_head/float(N), 4)
    bpp_pointsnums = round(8*bytes_pointnums/float(N) ,4)
    bpp_cubepos = round(8*bytes_cubepos/float(N), 4)
    bpps = [bpp, bpp_strings, bpp_strings_hyper, bpp_strings_head, bpp_pointsnums, bpp_cubepos]

    return cubes_d, cube_positions_d, points_numbers_d, N, bpps


def collect_results(results, bpps, N, scale, rho_d1, rho_d2):
    # bpp
    results["ori_points"] = N
    results["scale"] = scale
    # results["cube_size"] = cube_size
    # results["res"] = res
    results["bpp"] = bpps[0]
    results["bpp_strings"] = bpps[1]
    results["bpp_strings_hyper"] = bpps[2]
    results["bpp_strings_head"] = bpps[3]
    results["bpp_pointsnums"] = bpps[4]
    results["bpp_cubepos"] = bpps[5]

    results["rho_d1"] = rho_d1
    results["rho_d2"] = rho_d2

    print(results)

    return results

def plot_results(all_results, filename, root_dir):
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color = 'blue')

    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["optimal D1 PSNR"][:]), 
            label="D1 (optimal)", marker='h', color='red', linestyle='-.')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["optimal D2 PSNR"][:]), 
            label="D2 (optimal)", marker='h', color='blue', linestyle='-.')
    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(root_dir, filename+'.png'))

    return 

def eval(input_file, resultdir, res, ckpt_dir, cube_size, min_num, fixed_thres, postfix, batch_parallel):

    filename = os.path.split(input_file)[-1][:-4]
    input_file_n = input_file    
    csv_resultdir = resultdir
    if not os.path.exists(csv_resultdir):
        os.makedirs(csv_resultdir)
    csv_name = os.path.join(csv_resultdir, filename + '.csv')

    print('cube size:', cube_size, 'min num:', min_num, 'res:', res)

    scale = 1.0
    rho_d1 = 1.0
    rho_d2 = 1.0
    print('='*20, '\n', 'scale:', scale, 'ckpt_dir:', ckpt_dir, 'rho (d1):', rho_d1, 'rho_d2:', rho_d2)

    model = PCCModel(lower_bound=1e-9)
    ckpt = torch.load(ckpt_dir)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    cubes_d, cube_positions, points_numbers, N, bpps = test_hyper(input_file, resultdir, model, scale, cube_size, min_num, batch_parallel, postfix)
    cubes_d = cubes_d.numpy()
    print("bpp:",bpps[0])

    # metrics.
    rho = 1.0
    output_file = os.path.join(resultdir, filename + '_rec_' + '_' + 'rho' + str(round(rho*100)) + postfix + '.ply')
    postprocess(output_file, cubes_d, points_numbers, cube_positions, scale, cube_size, rho, fixed_thres)
    results = pc_error(input_file, output_file, input_file_n, res, show=False)

    results = collect_results(results, bpps, N, scale, rho_d1, rho_d2)
    all_results = results.copy(deep=True)
    all_results.to_csv(csv_name, index=False)
    print(all_results)
    

    return all_results

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=str, default='/media/avitech/Data/quocanhle/PointCloud/dataset/testdata/8iVFB/longdress_vox10_1300.ply', dest="input")
    parser.add_argument("--resultdir", type=str, default='/media/avitech/Data/quocanhle/PointCloud/results/PCGCv1/longdress_vox10_1300/', dest="resultdir")
    parser.add_argument("--res", type=int, default=1024, dest="res")
    #parser.add_argument("--mode", type=str, default='hyper', dest="mode")
    parser.add_argument("--ckpt_dir", type=str, default='/media/avitech/Data/quocanhle/PointCloud/pretrained_models/PCGCv1/a10/epoch_29_2999.pth', dest="ckpt_dir",  help='checkpoint')
    parser.add_argument("--cube_size", type=int, default=64, dest="cube_size")
    parser.add_argument("--min_num", type=int, default=64, dest="min_num")
    #parser.add_argument("--modelname", default="models.model_voxception", help="(model_simple, model_voxception)", dest="modelname")
    parser.add_argument("--fixed_thres", type=float, default=None, help="fixed threshold ", dest="fixed_thres")    
    parser.add_argument("--postfix", default="", help="", dest="postfix") 
    parser.add_argument("--batch_parallel", type=int, default=16, dest="batch_parallel", help="size of parallel batches,depend on gpu memory.") #并行batch大小，取决于显存，大了更快，但太大显存会爆
   
    args = parser.parse_args()
    print(args)
    return args

if __name__ == "__main__":
    args = parse_args()
    if not os.path.exists(args.resultdir):
        os.makedirs(args.resultdir)
        
    print(args.input)
    all_results = eval(args.input, args.resultdir, args.res, args.ckpt_dir, 
                    args.cube_size, args.min_num, args.fixed_thres, args.postfix, args.batch_parallel)

    """
    python eval.py --input "testdata/8iVFB/longdress_vox10_1300.ply" \
                    --resultdir="results/hyper/" \
                    --cfgdir="results/hyper/8iVFB_vox10.ini" \
                    --res=1024
    """
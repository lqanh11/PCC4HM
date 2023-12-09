# Copyright (c) Nanjing University, Vision Lab.
# Last update:
# 2021.9.9

import os
import argparse
import numpy as np
import torch
import time
import importlib 
from pcc_model import PCCModel

from process import preprocess, postprocess
from transform import compress_hyper, decompress_hyper
from dataprocess.inout_bitstream import write_binary_files_hyper, read_binary_files_hyper

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--command", default="decompress", #"decompress" 'compress'
                        help="What to do: 'compress' reads a point cloud (.ply format) "
                            "and writes compressed binary files. 'decompress' "
                            "reads binary files and reconstructs the point cloud (.ply format). "
                            "input and output filenames need to be provided for the latter. ")
    parser.add_argument("--input", default='compressed/redandblack_vox10_1550', nargs="?", help="Input filename.") #'compressed/28_airplane_0270' '/userhome/PCGCv1/testdata/8iVFB/redandblack_vox10_1550.ply'
    parser.add_argument("output", nargs="?", help="Output filename.")
    parser.add_argument("--ckpt_dir", type=str, default='/userhome/PCGCv1/pytorch2/ckpts/hyper_mgpu2/epoch_13_12599.pth', dest="ckpt_dir",  help='checkpoint')
    parser.add_argument("--scale", type=float, default=1.0, dest="scale", help="scaling factor.")
    parser.add_argument("--cube_size", type=int, default=64, dest="cube_size", help="size of partitioned cubes.") 
    parser.add_argument("--min_num", type=int, default=64, dest="min_num", help="minimum number of points in a cube.")
    parser.add_argument("--rho", type=float, default=1.0, dest="rho", help="ratio of the numbers of output points to the number of input points.")
    parser.add_argument("--gpu", type=int, default=1, dest="gpu", help="use gpu (1) or not (0).") 
    parser.add_argument("--batch_parallel", type=int, default=64, dest="batch_parallel", help="size of parallel batches,depend on gpu memory.") 
    args = parser.parse_args()
    print(args)

    return args

if __name__ == "__main__":
    """
    Examples:
    python test.py compress "testdata/8iVFB/longdress_vox10_1300.ply" \
        --ckpt_dir="checkpoints/hyper/a6b3/" 
    python test.py decompress "compressed/longdress_vox10_1300" \
        --ckpt_dir="checkpoints/hyper/a6b3/"
    """
    args = parse_args()
    if args.gpu==1:
        os.environ['CUDA_VISIBLE_DEVICES']="0"
    else:
        os.environ['CUDA_VISIBLE_DEVICES']=""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    model = PCCModel(lower_bound=1e-9)
    ckpt = torch.load(args.ckpt_dir)
    model.load_state_dict(ckpt['model'])
    model.to(device)

    if args.command == "compress":
        if not args.output:
            args.output = os.path.split(args.input)[-1][:-4]
        cubes, cube_positions, points_numbers = preprocess(args.input, args.scale, args.cube_size, args.min_num)
        y_strings, y_min_vs, y_max_vs, y_shape, z_strings, z_min_v, z_max_v, z_shape = compress_hyper(cubes, model, device, args.batch_parallel)

        bytes_strings, bytes_strings_head, bytes_strings_hyper, bytes_pointnums, bytes_cubepos = write_binary_files_hyper(
            args.output, y_strings, z_strings, points_numbers, cube_positions,
            y_min_vs, y_max_vs, y_shape.numpy(), 
            z_min_v, z_max_v, z_shape.numpy(), rootdir='./compressed')

    elif args.command == "decompress":
        rootdir, filename = os.path.split(args.input)
        if not args.output:
            args.output = filename + "_rec.ply"

        y_strings_d, z_strings_d, points_numbers_d, cube_positions_d, \
        y_min_vs_d, y_max_vs_d, y_shape_d, z_min_v_d, z_max_v_d, z_shape_d = read_binary_files_hyper(filename, rootdir)
        cubes_d = decompress_hyper(y_strings_d, y_min_vs_d, y_max_vs_d, y_shape_d, z_strings_d, z_min_v_d, z_max_v_d, z_shape_d, model,device,args.batch_parallel)
        postprocess(args.output, cubes_d.numpy(), points_numbers_d, cube_positions_d, args.scale, args.cube_size, args.rho)
        
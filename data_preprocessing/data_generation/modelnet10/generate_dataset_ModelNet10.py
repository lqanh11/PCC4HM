import sys

import argparse
import open3d as o3d
import numpy as np
import random
import os, time
from data_utils import write_ply_ascii_geo
from pathlib import Path
from pyntcloud import PyntCloud

import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

def process_points(args, points):
    assert points.ndim == 2 and points.shape[-1] == 3
    resolution_array = []

    if args.normalize:
        points = points - points.mean(axis=0)
        point_norms = np.sqrt((points**2).sum(axis=1))
        points = points / point_norms.max()

    if args.translate != [0, 0, 0]:
        points = points + np.array(args.translate)

    if args.scale != 1.0:
        points = points * args.scale

    if args.resolution is not None:
        # normalize to fixed resolution.
        x_max = np.max(points[:,0])
        x_min = np.min(points[:,0])
        x_length = x_max - x_min

        y_max = np.max(points[:,1])
        y_min = np.min(points[:,1])
        y_length = y_max - y_min

        z_max = np.max(points[:,2])
        z_min = np.min(points[:,2])
        z_length = z_max - z_min

        length_array = [x_length, y_length, z_length]
        idx_max = np.argmax(length_array)

        resolution_array = [args.resolution, args.resolution, args.resolution]
        resolution_array[idx_max-1] = np.ceil(resolution_array[idx_max-1] * (length_array[idx_max-1]/length_array[idx_max]))
        resolution_array[idx_max-2] = np.ceil(resolution_array[idx_max-2] * (length_array[idx_max-2]/length_array[idx_max]))

        for axis_index in range(np.shape(points)[1]):
            points[:,axis_index] = points[:,axis_index] - np.min(points[:,axis_index])
            points[:,axis_index] = points[:,axis_index] / np.max(points[:,axis_index])
            points[:,axis_index] = points[:,axis_index] * (resolution_array[axis_index])
 
    if args.quantize:
        points = points.round()

    return points, resolution_array


def process_cloud(args, cloud):
    xyz = ["x", "y", "z"]
    assert cloud.points.columns.tolist() == xyz

    mesh = cloud.mesh
    cloud.points = cloud.points.astype(np.float64, copy=False)
    cloud.mesh = mesh  # Restore mesh.

    if args.num_points_resample > 0:
        cloud = cloud.get_sample(
            "mesh_random", n=args.num_points_resample, as_PyntCloud=True
        )

    cloud.points[xyz], resolution_array = process_points(args, cloud.points[xyz].values)
    if args.unique:
        _, idx = np.unique(cloud.points[xyz].values, axis=0, return_index=True)
        cloud.points = cloud.points.iloc[idx]

    if args.num_points_subsample > 0:
        # print(len(cloud.points))
        assert len(cloud.points) >= args.num_points_subsample
        idx = np.random.choice(
            len(cloud.points), args.num_points_subsample, replace=False
        )
        cloud.points = cloud.points.iloc[idx]

    return cloud.points[xyz].values, resolution_array

def iterate_paths(args):

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    for input_filepath in input_root.rglob(f"*.{args.input_extension}"):
        filepath = Path(input_filepath).relative_to(input_root)
        output_filepath = (output_root / filepath).with_suffix(
            f".{args.output_extension}"
        )
        output_filepath.parent.mkdir(parents=True, exist_ok=True)
        yield input_filepath, output_filepath

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/mesh_original_format_off") # path of the OFF file in ModelNet
    parser.add_argument("--output_dir", type=str, default="/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_ply/") # save folder directory
    parser.add_argument("--input_extension", type=str, default="off")
    parser.add_argument("--output_extension", type=str, default="ply")

    parser.add_argument("--normalize", type=bool, default=False) # normalize the coordinate to zero mean 
    parser.add_argument("--translate", type=float, nargs=3, default=[0, 0, 0])
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=128) # normalize the coordinate into the range [0, resolution] # default: 2048
    parser.add_argument("--quantize", type=bool, default=True)
    parser.add_argument("--unique", type=bool, default=True)
    parser.add_argument("--num_points_resample", type=int, default=500000)
    parser.add_argument("--num_points_subsample", type=int, default=0)

    parser.add_argument("--array_list", default=None)
    

    return parser


def parse_args(parser, argv=None):
    args = parser.parse_args(argv)
    return args

def process(index, args):
    print(index)
    input_filepath, output_filepath = args.array_list[index]
    # print(f"Processing {index}\n  in:  {input_filepath}\n  out: {output_filepath}")

    try:
        cloud = PyntCloud.from_file(str(input_filepath))
        points, resolution_array = process_cloud(args, cloud)
        
        if 0 in resolution_array:
            print(input_filepath)
        else:
            write_ply_ascii_geo(str(output_filepath), points)

    except:
        print(input_filepath)
    return

if __name__ == "__main__":
    

    parser = build_parser()
    args = parse_args(parser)
    args.output_dir = os.path.join(args.output_dir, f'{args.resolution}_dense')
    ##################################################################################3
  
    array_file = []
    # count = 0

    for input_filepath, output_filepath in sorted(iterate_paths(args)): 
        # if count < 5:
        array_file.append([input_filepath, output_filepath])
        # count += 1

    args.array_list = array_file

    from multiprocessing import Pool
    import functools

    with Pool(15) as p:
        process_f = functools.partial(process, args=args)
        tqdm(p.map(process_f, range(len(array_file))), total=len(array_file))
    
    print('DONE!'*10)

    
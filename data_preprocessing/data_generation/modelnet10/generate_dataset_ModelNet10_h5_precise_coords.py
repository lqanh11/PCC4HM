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
import h5py
import copy

def process_points(points, resolution):
    assert points.ndim == 2 and points.shape[-1] == 3
    resolution_array = []

    normalize = True
    if normalize:
        points = points - points.mean(axis=0)
        point_norms = np.sqrt((points**2).sum(axis=1))
        points = points / point_norms.max()

    if resolution is not None:
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

        resolution_array = [resolution, resolution, resolution]
        resolution_array[idx_max-1] = np.ceil(resolution_array[idx_max-1] * (length_array[idx_max-1]/length_array[idx_max]))
        resolution_array[idx_max-2] = np.ceil(resolution_array[idx_max-2] * (length_array[idx_max-2]/length_array[idx_max]))

        for axis_index in range(np.shape(points)[1]):
            points[:,axis_index] = points[:,axis_index] - np.min(points[:,axis_index])
            points[:,axis_index] = points[:,axis_index] / np.max(points[:,axis_index])
            points[:,axis_index] = points[:,axis_index] * (resolution_array[axis_index])
    
    quantize = True
    if quantize:
        points = points.round()

    return points, resolution_array

def process_cloud(cloud, num_points_resample):
    xyz = ["x", "y", "z"]
    assert cloud.points.columns.tolist() == xyz

    mesh = cloud.mesh
    cloud.points = cloud.points.astype(np.float64, copy=False)
    cloud.mesh = mesh  # Restore mesh.

    if num_points_resample > 0:
        cloud = cloud.get_sample(
            "mesh_random", n=num_points_resample, as_PyntCloud=True
        )

    return cloud

if __name__ == "__main__":

    root_data = '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10'

    split_path = os.path.join(root_data, 'split')
    catfile = os.path.join(split_path,'modelnet10_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))

    shape_ids = {}
    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(split_path, 'modelnet10_train.txt'))]
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(split_path, 'modelnet10_test.txt'))]

    for split in ['train', 'test']:
        ## save_file_path
        save_h5_dir = os.path.join(root_data, 'pc_modelnet10_ply_hdf5_single_file', f'precise_coordinates', split)
        os.makedirs(save_h5_dir, exist_ok=True)
                    
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

        datapath = [(shape_names[i], os.path.join(
            root_data,
            'mesh_original_format_off',
            shape_names[i],
            split,
            shape_ids[split][i]) + '.off',
            shape_ids[split][i] + '.h5'
            ) for i in range(len(shape_ids[split]))]
        
        print('The size of %s data is %d' % (split, len(datapath)))

        list_of_points = [None] * len(datapath)
        list_of_labels = [None] * len(datapath)

        for index in tqdm(range(len(datapath)), total=len(datapath)):
            fn = datapath[index]
            cls = classes[datapath[index][0]]
            cls = np.array([cls]).astype(np.uint8)

            cloud = PyntCloud.from_file(fn[1])
            resampled_cloud = process_cloud(cloud, 40000)

            h5_file_path = os.path.join(save_h5_dir, fn[2])

            if os.path.exists(h5_file_path): os.remove(h5_file_path)
            save_h5_file = h5py.File(h5_file_path, 'w')
            save_h5_file.create_dataset("label", data = cls)

            process_resampled_cloud = copy.deepcopy(resampled_cloud)

            for num_points_subsample in [512, 1024, 2048, 4096, 8192, 16384, 32768]:
                idx = np.random.choice(
                    len(resampled_cloud.points), num_points_subsample, replace=False
                )
                process_resampled_cloud.points = resampled_cloud.points.iloc[idx]
                
                coords = process_resampled_cloud.points[["x", "y", "z"]].values.astype(np.float32)

                save_h5_file.create_dataset(f"data_{num_points_subsample}", data = coords)
                
            
        
    
    
    
    
    # args.output_dir = os.path.join(args.output_dir, f'dense_{args.resolution}')
    # ##################################################################################3
  
    # array_file = []
    # # count = 0

    # for input_filepath, output_filepath in sorted(iterate_paths(args)): 
    #     # if count < 5:
    #     array_file.append([input_filepath, output_filepath])
    #     # count += 1

    # args.array_list = array_file

    # from multiprocessing import Pool
    # import functools

    # with Pool(15) as p:
    #     process_f = functools.partial(process, args=args)
    #     tqdm(p.map(process_f, range(len(array_file))), total=len(array_file))
    
    # print('DONE!'*10)

    
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
    
    points = points.round()
    points = np.unique(points, axis=0)

    return points

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

    points = {}
    points_512 = {}
    points_1024 = {}
    points_2048 = {}

    for resolution in [64, 128, 256, 512, 1024]:
        points[resolution] = process_points(cloud.points[xyz].values, resolution)      

        list_idx = {}
        for num_points_subsample in [512, 1024, 2048]:
            if len(points[resolution]) >= num_points_subsample:
                idx = np.random.choice(
                    len(points[resolution]), num_points_subsample, replace=False
                )
                list_idx[num_points_subsample] = idx
            else:
                missing_points = num_points_subsample - len(points[resolution])
                idx = np.random.choice(
                    len(points[resolution]), missing_points, replace=False
                )
                list_idx = np.array(list(range(len(cloud.points)))) 
                new_idx = np.concatenate((list_idx, idx))
                list_idx[num_points_subsample] = new_idx

        points_512[resolution] = points[resolution][list_idx[512]]
        points_1024[resolution] = points[resolution][list_idx[1024]]
        points_2048[resolution] = points[resolution][list_idx[2048]]

    return points, points_512, points_1024, points_2048

if __name__ == "__main__":

    root_data = '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10'
    save_h5_path = '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/'

    split_path = os.path.join(root_data, 'split')
    catfile = os.path.join(split_path,'modelnet10_shape_names.txt')
    cat = [line.rstrip() for line in open(catfile)]
    classes = dict(zip(cat, range(len(cat))))

    shape_ids = {}
    shape_ids['train'] = [line.rstrip() for line in open(os.path.join(split_path, 'modelnet10_train.txt'))]
    shape_ids['test'] = [line.rstrip() for line in open(os.path.join(split_path, 'modelnet10_test.txt'))]

    ## create h5 for all files
    shape_ids['all'] = shape_ids['train'] + shape_ids['test']
    shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids['all']]

    datapath = [(shape_names[i], 
                 os.path.join(root_data,
                              'mesh_original_format_off',
                              shape_names[i],
                              'train' if shape_ids['all'][i] in shape_ids['train'] else 'test',
                              shape_ids['all'][i]) + '.off', 
                 os.path.join(save_h5_path,
                              'all_resolution',
                              'train' if shape_ids['all'][i] in shape_ids['train'] else 'test',
                              shape_ids['all'][i]) + '.h5'
                ) for i in range(len(shape_ids['all']))]
    
    for index in tqdm(range(len(datapath)), total=len(datapath)):
        fn = datapath[index]
        cls = classes[datapath[index][0]]
        cls = np.array([cls]).astype(np.uint8)

        h5_file_path = fn[2]
        if os.path.exists(h5_file_path): os.remove(h5_file_path)
        save_h5_file = h5py.File(h5_file_path, 'w')

        # read mesh file
        cloud = PyntCloud.from_file(fn[1])
        # mesh to point cloud process
        points, points_512, points_1024, points_2048 = process_cloud(cloud, 500000)
        
        for resolution in [64, 128, 256, 512, 1024]:
            save_h5_file.create_dataset(f"points_{resolution}", data = points[resolution])
            save_h5_file.create_dataset(f"points_512_{resolution}", data = points_512[resolution])
            save_h5_file.create_dataset(f"points_1024_{resolution}", data = points_1024[resolution])
            save_h5_file.create_dataset(f"points_2048_{resolution}", data = points_2048[resolution])
        save_h5_file.create_dataset(f"class", data = cls)
        save_h5_file.create_dataset(f"class_name", data = fn[0])

    # for resolution in [64, 128, 256, 512, 1024]:
    #     for num_points_subsample in [1024, 2048]:
    #         for split in ['train', 'test']:
    #             ## save_file_path
    #             save_ply_dir = os.path.join(root_data, f'pc_modelnet10_ply_resampled_{num_points_subsample}', f'voxelized_{resolution}')
    #             os.makedirs(save_ply_dir, exist_ok=True)

    #             save_h5_dir = os.path.join(root_data, f'pc_modelnet10_ply_hdf5_{num_points_subsample}', f'voxelized_{resolution}')
    #             os.makedirs(save_h5_dir, exist_ok=True)

    #             h5_file_path = os.path.join(save_h5_dir, f'ply_data_{split}_vox007.h5')
    #             if os.path.exists(h5_file_path): os.remove(h5_file_path)
    #             save_h5_file = h5py.File(h5_file_path, 'w')
                
                
    #             shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]

    #             datapath = [(shape_names[i], os.path.join(
    #                 root_data,
    #                 'mesh_original_format_off',
    #                 shape_names[i],
    #                 split,
    #                 shape_ids[split][i]) + '.off') for i in range(len(shape_ids[split]))]
                
    #             print('The size of %s data is %d' % (split, len(datapath)))

    #             list_of_points = [None] * len(datapath)
    #             list_of_labels = [None] * len(datapath)

    #             for index in tqdm(range(len(datapath)), total=len(datapath)):
    #                 fn = datapath[index]
    #                 cls = classes[datapath[index][0]]
    #                 cls = np.array([cls]).astype(np.uint8)

    #                 cloud = PyntCloud.from_file(fn[1])
    #                 point_set = process_cloud(cloud, 40000, num_points_subsample, resolution).astype(np.float32)
                    
    #                 list_of_points[index] = point_set
    #                 list_of_labels[index] = cls

    #                 ply_shapename_dir = os.path.join(save_ply_dir, fn[0], split)
    #                 os.makedirs(ply_shapename_dir, exist_ok=True)
    #                 ply_file_path = os.path.join(ply_shapename_dir, fn[1].split('/')[-1].replace('.off', '.ply'))

    #                 write_ply_ascii_geo(ply_file_path, point_set)

    #             save_h5_file.create_dataset("data", data = list_of_points)
    #             save_h5_file.create_dataset("label", data = list_of_labels)
                
        
    
    
    
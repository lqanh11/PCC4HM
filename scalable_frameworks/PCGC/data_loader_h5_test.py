import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_utils import read_h5_geo_read_all_info
from data_utils import read_h5_geo_read_all_info, read_ply_ascii_geo, array2vector
from gpcc import gpcc_decode
import copy

class InfSampler(Sampler):
    """Samples elements randomly, without replacement.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source, shuffle=False):
        self.data_source = data_source
        self.shuffle = shuffle
        self.reset_permutation()

    def reset_permutation(self):
        perm = len(self.data_source)
        if self.shuffle:
            perm = torch.randperm(perm)
        self._perm = perm.tolist()

    def __iter__(self):
        return self

    def __next__(self):
        if len(self._perm) == 0:
            self.reset_permutation()
        return self._perm.pop()

    def __len__(self):
        return len(self.data_source)

def latentspace_collate_fn(list_data):
    sparse_xyz_batch, sparse_features_batch, labels_batch = ME.utils.sparse_collate(
        [d["sparse_coordinates"] for d in list_data],
        [d["sparse_features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    num_points_batch = [d["num_points"] for d in list_data]
    dense_pc_file_dir_batch = [d["dense_pc_file_dir"] for d in list_data]
    compression_files_dir_batch = [d["compression_files_dir"] for d in list_data]

    return {
        "sparse_coordinates": sparse_xyz_batch.to(torch.float32),
        "sparse_features": sparse_features_batch.to(torch.float32),

        "labels": labels_batch,
        'num_points': num_points_batch,
        "dense_pc_file_dir": dense_pc_file_dir_batch,
        "compression_files_dir": compression_files_dir_batch
    }

def minkowski_collate_fn(list_data):
    dense_xyz_batch, dense_features_batch, labels_batch = ME.utils.sparse_collate(
        [d["dense_coordinates"] for d in list_data],
        [d["dense_features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    sparse_xyz_batch, sparse_features_batch, labels_batch = ME.utils.sparse_collate(
        [d["sparse_coordinates"] for d in list_data],
        [d["sparse_features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )

    return {
        "dense_coordinates": dense_xyz_batch.to(torch.float32),
        "dense_features": dense_features_batch.to(torch.float32),

        "sparse_coordinates": sparse_xyz_batch.to(torch.float32),
        "sparse_features": sparse_features_batch.to(torch.float32),

        "labels": labels_batch,
    }
    

def collate_pointcloud_fn(list_data):
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    
    dense_xyz_batch, dense_features_batch, labels_batch = ME.utils.sparse_collate(
        [d["dense_coordinates"] for d in list_data],
        [d["dense_features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    sparse_xyz_batch, sparse_features_batch, labels_batch = ME.utils.sparse_collate(
        [d["sparse_coordinates"] for d in list_data],
        [d["sparse_features"] for d in list_data],
        [d["label"] for d in list_data],
        dtype=torch.float32,
    )
    dense_pc_file_dir_batch = [d["dense_pc_file_dir"] for d in list_data]

    return {
        "dense_coordinates": dense_xyz_batch.to(torch.float32),
        "dense_features": dense_features_batch.to(torch.float32),

        "sparse_coordinates": sparse_xyz_batch.to(torch.float32),
        "sparse_features": sparse_features_batch.to(torch.float32),

        "dense_pc_file_dir": dense_pc_file_dir_batch,
        "labels": labels_batch,
    }

class PCDataset_LoadAll                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (torch.utils.data.Dataset):

    def __init__(self, files, resolution=128, num_pts=1024):
        self.files = files
        self.resolution = resolution
        self.num_pts = num_pts

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        basename = os.path.split(filedir)[-1].split('.')[0]
        
        ply_file_dir = os.path.join('/media/avitech/QuocAnh_1TB/Point_Cloud/dataset/modelnet10/pc_resample_format_ply',
                                    f'{self.resolution}',
                                    'original',
                                    basename + ".ply")

        data = read_h5_geo_read_all_info(filedir)
        coords = data[f"points_{self.resolution}"][:].astype("float32")
        coords_fix_pts = data[f"points_{self.num_pts}_{self.resolution}"][:].astype("float32")

        feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype("float32")
        feats_fix_pts = np.expand_dims(np.ones(coords_fix_pts.shape[0]), 1).astype("float32")

        # object labels
        label = data["class"][:].astype("int64")
        label = torch.from_numpy(label)

        return {
            "dense_pc_file_dir": ply_file_dir,

            "dense_coordinates": coords,
            "dense_features": feats,

            "sparse_coordinates": coords_fix_pts,
            "sparse_features": feats_fix_pts,

            "label": label,
        }
        

def make_data_loader(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader

list_test = [
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0117.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0128.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0130.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0126.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0147.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0590.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0550.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0534.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0605.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bed_0603.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0927.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0896.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0989.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0934.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/chair_0936.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0205.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0283.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0258.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0277.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/desk_0217.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0254.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0244.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0206.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0252.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/dresser_0219.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0497.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0481.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0526.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0556.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/monitor_0512.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0280.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0247.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0226.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0202.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/night_stand_0285.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0696.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0765.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0735.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0689.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/sofa_0733.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0472.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0466.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0443.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0451.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/table_0446.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0429.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0404.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0443.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0375.h5',
    '/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/toilet_0360.h5',
]


class ModelNet_latentspace(Dataset):
    def __init__(
        self,
        entropy_model,
        phase: str,
        data_root: str,
        resolution=128,
        rate='r7',
    ):
        Dataset.__init__(self)
        # download_modelnet40_dataset()
        phase = "test" if phase in ["val", "test"] else "train"
        self.data_root = data_root
        self.phase = phase
        self.phase = phase
        self.resolution = resolution
        self.rate = rate
        # self.files = glob.glob(os.path.join(data_root, phase, f'{self.resolution}', 'original', "*.ply"))
        self.files = list_test
        self.entropy_model = entropy_model.cpu()

    def __getitem__(self, i: int) -> dict:
        label_dict = {
            'bathtub':0,
            'bed':1,
            'chair':2,
            'desk':3,
            'dresser':4,
            'monitor':5,
            'night_stand':6,
            'sofa':7,
            'table':8,
            'toilet':9
        }

        ## process latent space
        basename = os.path.split(self.files[i])[-1].split('.')[0]

        ply_file_dir = os.path.join(self.data_root, self.phase, f'{self.resolution}', 'original', basename + ".ply")
        
        output_resolution = os.path.join(self.data_root, self.phase, f'{self.resolution}')
        output_rate = os.path.join(output_resolution, self.rate)

        gpcc_decode(os.path.join(output_rate,
                                 basename + '_C.bin'),
                    os.path.join(output_rate,
                                 basename + '_C.ply'))
        coords = read_ply_ascii_geo(os.path.join(output_rate, basename + '_C.ply'))
        coords = torch.cat((torch.zeros((len(coords),1)).int(), torch.tensor(coords).int()), dim=-1)
        indices_sort = np.argsort(array2vector(coords, coords.max()+1))
        coords = coords[indices_sort]
        coords = coords[:,1:]

        with open(os.path.join(output_rate, basename + '_F.bin'), 'rb') as fin:
            strings = fin.read()
        with open(os.path.join(output_rate, basename +'_H.bin'), 'rb') as fin:
            shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            len_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            min_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
            max_v = np.frombuffer(fin.read(4*len_min_v), dtype=np.float32)[0]
        with open(os.path.join(output_rate, basename +'_num_points.bin'), 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            num_points[-1] = int(num_points[-1])# update
            num_points = [num for num in num_points]
            
        feats = self.entropy_model.decompress(strings, min_v, max_v, shape, channels=shape[-1])

        # sparse_xyz = coords
        label = np.array([label_dict[basename[:len(basename) - len(basename.split('_')[-1]) - 1]]])

        ## process for sparse PCs
        # sparse_xyz_torch = torch.from_numpy(sparse_xyz)
        sparse_xyz_torch = coords
        sparse_features_torch = feats
        
        label = torch.from_numpy(label)

        return {
            # "dense_pc_file_dir": self.files[i],
            "dense_pc_file_dir": ply_file_dir,
            "compression_files_dir": [
                                        os.path.join(output_rate, basename + '_C.bin'), 
                                        os.path.join(output_rate, basename + '_F.bin'), 
                                        os.path.join(output_rate, basename + '_H.bin'), 
                                        os.path.join(output_rate, basename + '_num_points.bin')                     
                                      ],

            "sparse_coordinates": sparse_xyz_torch.to(torch.float32),
            "sparse_features": sparse_features_torch.to(torch.float32),

            "label": label,
            'num_points': num_points,
        }

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f"ModelNetH5_voxelize_all(phase={self.phase}, length={len(self)}, transform={self.transform})"

def make_data_loader_latentspace(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=latentspace_collate_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True,
        'drop_last': False
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)
    return loader


# if __name__ == "__main__":
#     # filedirs = sorted(glob.glob('/home/ubuntu/HardDisk2/color_training_datasets/training_dataset/'+'*.h5'))
#     filedirs = sorted(glob.glob('/home/ubuntu/HardDisk1/point_cloud_testing_datasets/8i_voxeilzaed_full_bodies/8i/longdress/Ply/'+'*.ply'))
#     test_dataset = PCDataset(filedirs[:10])
#     test_dataloader = make_data_loader(dataset=test_dataset, batch_size=2, shuffle=True, num_workers=1, repeat=False,
#                                         collate_fn=collate_pointcloud_fn)
#     for idx, (coords, feats) in enumerate(tqdm(test_dataloader)):
#         print("="*20, "check dataset", "="*20, 
#             "\ncoords:\n", coords, "\nfeat:\n", feats)

#     test_iter = iter(test_dataloader)
#     print(test_iter)
#     for i in tqdm(range(10)):
#         coords, feats = test_iter.next()
#         print("="*20, "check dataset", "="*20, 
#             "\ncoords:\n", coords, "\nfeat:\n", feats)




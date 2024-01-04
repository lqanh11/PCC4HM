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
    # new_list_data = []
    # num_removed = 0
    # for data in list_data:
    #     if data is not None:
    #         new_list_data.append(data)
    #     else:
    #         num_removed += 1
    # list_data = new_list_data
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    coords, coords_fix_pts, feats, feats_fix_pts, labels = list(zip(*list_data))
    coords_batch, feats_batch, labels_batch = ME.utils.sparse_collate(coords,
                                                                      feats, 
                                                                      labels, 
                                                                      dtype=torch.float32)
    coords_fix_pts_batch, feats_fix_pts_batch, labels_batch = ME.utils.sparse_collate(coords_fix_pts,
                                                                      feats_fix_pts, 
                                                                      labels, 
                                                                      dtype=torch.float32)

    return coords_batch, coords_fix_pts_batch, feats_batch, feats_fix_pts_batch, labels_batch

class PCDataset_LoadAll                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (torch.utils.data.Dataset):

    def __init__(self, files, resolution=128, num_pts=1024):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files
        self.resolution = resolution
        self.num_pts = num_pts

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]
        
        if idx in self.cache:
            coords, coords_fix_pts, feats, feats_fix_pts, labels = self.cache[idx]
        else:
            data = read_h5_geo_read_all_info(filedir)
            coords = data[f"points_{self.resolution}"][:].astype("float32")
            coords_fix_pts = data[f"points_{self.num_pts}_{self.resolution}"][:].astype("float32")

            feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype("float32")

            # normalize coordinate for feature
            
            # xyz = coords_fix_pts - coords_fix_pts.mean(axis=0)
            # xyz_norm = np.sqrt((xyz**2).sum(axis=1))
            # feats_fix_pts = xyz / xyz_norm.max()

            feats_fix_pts = np.expand_dims(np.ones(coords_fix_pts.shape[0]), 1).astype("float32")

            # object labels
            labels = data["class"][:].astype('int64')

            # cache
            self.cache[idx] = (coords, coords_fix_pts, feats, feats_fix_pts, labels)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent

        return (coords, coords_fix_pts, feats, feats_fix_pts, labels)


class ModelNetH5_voxelize_all(Dataset):
    def __init__(
        self,
        phase: str,
        data_root: str = "modelnet40h5",
        transform=None,
        num_points=1024,
        resolution=128
    ):
        Dataset.__init__(self)
        # download_modelnet40_dataset()
        phase = "test" if phase in ["val", "test"] else "train"
        self.transform = transform
        self.phase = phase
        self.num_points = num_points
        self.resolution = resolution
        self.files = glob.glob(os.path.join(data_root, phase, "*.h5"))

        # self.data, self.data_fix_pts, self.label = self.load_data(data_root, phase)
        

    # def load_data(self, data_root, phase):
    #     data, data_fix_pts, labels = [], [], []

    #     assert os.path.exists(data_root), f"{data_root} does not exist"
    #     files = glob.glob(os.path.join(data_root, phase, "*.h5"))
    #     assert len(files) > 0, "No files found"
    #     for h5_name in files:
    #         with h5py.File(h5_name) as f:
    #             data.append(f[f"points_{self.resolution}"][:].astype("float32"))
    #             data_fix_pts.append(f[f"points_{self.num_points}_{self.resolution}"][:].astype("float32"))
    #             labels.append(f["class"][:].astype("int64"))

    #     return data, data_fix_pts, labels

    def __getitem__(self, i: int) -> dict:
        
        h5_name = self.files[i]
        with h5py.File(h5_name) as f:
            dense_xyz = f[f"points_{self.resolution}"][:].astype("float32")
            sparse_xyz = f[f"points_{self.num_points}_{self.resolution}"][:].astype("float32")
            label = f["class"][:].astype("int64")
        
        # dense_xyz = self.data[i]

        # sparse_xyz = self.data_fix_pts[i]
        if self.phase == "train":
            np.random.shuffle(sparse_xyz)
            np.random.shuffle(dense_xyz)

        if len(sparse_xyz) > self.num_points:
            sparse_xyz = sparse_xyz[: self.num_points]
        if self.transform is not None:
            dense_xyz = self.transform(dense_xyz)
            sparse_xyz = self.transform(sparse_xyz)

        # label = self.label[i]
        ## process for dense PCs
        dense_xyz_torch = torch.from_numpy(dense_xyz)
        dense_features_torch = torch.from_numpy(np.expand_dims(np.ones(dense_xyz.shape[0]), 1))

        ## process for sparse PCs
        # coordinates
        sparse_xyz_torch = torch.from_numpy(sparse_xyz)

        # features
        sparse_features = copy.deepcopy(sparse_xyz)
        sparse_features = sparse_features - sparse_features.mean(axis=0)
        sparse_features_norm = np.sqrt((sparse_features**2).sum(axis=1))
        sparse_features = sparse_features / sparse_features_norm.max()
        sparse_features_torch = torch.from_numpy(sparse_features)
        
        label = torch.from_numpy(label)

        return {
            "dense_coordinates": dense_xyz_torch.to(torch.float32),
            "dense_features": dense_features_torch.to(torch.float32),

            "sparse_coordinates": sparse_xyz_torch.to(torch.float32),
            "sparse_features": sparse_features_torch.to(torch.float32),

            "label": label,
        }

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return f"ModelNetH5_voxelize_all(phase={self.phase}, length={len(self)}, transform={self.transform})"

def make_data_loader_minkowski(dataset, batch_size=1, shuffle=True, num_workers=1, repeat=False, 
                    collate_fn=minkowski_collate_fn):
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
        self.files = glob.glob(os.path.join(data_root, phase, f'{self.resolution}', 'original', "*.ply"))
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
            "dense_pc_file_dir": self.files[i],
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




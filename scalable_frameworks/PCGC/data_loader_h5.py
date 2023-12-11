import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
import h5py
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
import MinkowskiEngine as ME
from data_utils import read_h5_geo_read_all_info

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


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data
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
            coords = data[f"points_{self.resolution}"][:].astype('int')
            coords_fix_pts = data[f"points_{self.num_pts}_{self.resolution}"][:].astype('int')

            feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            feats_fix_pts = np.expand_dims(np.ones(coords_fix_pts.shape[0]), 1).astype('int')

            # object labels
            labels = data["class"][:].astype('int64')

            # cache
            self.cache[idx] = (coords, coords_fix_pts, feats, feats_fix_pts, labels)
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        feats = feats.astype("float32")

        return (coords, coords_fix_pts, feats, feats_fix_pts, labels)


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




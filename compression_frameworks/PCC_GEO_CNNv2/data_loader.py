import os, sys, glob
import time
from tqdm import tqdm
import numpy as np
# import h5py
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler
from pyntcloud import PyntCloud

class InfSampler(Sampler):#不用改
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

#已改
def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1
    list_data = new_list_data #去掉了空值
    if len(list_data) == 0:
        raise ValueError('No data in the batch')
    #coords, feats = list(zip(*list_data)) #将list中的元组中的每一项取出,各自添加到一起,组成新的list
    #coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)

    return list_data #coords_batch, feats_batch

def points2voxels(set_points, cube_size):
  """Transform points to voxels (binary occupancy map).
  Args: points list; cube size;

  Return: A tensor with shape [batch_size, cube_size, cube_size, cube_size, 1]
  """

  voxels = []
  for _, points in enumerate(set_points):
    points = points.astype("int")
    vol = np.zeros((cube_size,cube_size,cube_size))
    vol[points[:,0],points[:,1],points[:,2]] = 1.0
    # vol = np.expand_dims(vol,0) 
    voxels.append(vol)
  voxels = np.array(voxels)

  return voxels

#已改
class PCDataset(torch.utils.data.Dataset):

    def __init__(self, files, resolution=64):
        self.files = []
        self.cache = {}
        self.last_cache_percent = 0
        self.files = files
        self.resolution = resolution

    def __len__(self):

        return len(self.files)
    
    def __getitem__(self, idx):
        filedir = self.files[idx]

        if idx in self.cache:
            points = self.cache[idx]
        else:
            # points = h5py.File(filedir, 'r')['data'][:].astype('int') #这个数据集只有xyz
            pc = PyntCloud.from_file(filedir)
            points = pc.points[['x','y','z']].values #(5616, 3) [50. 47. 62.]
            points = points2voxels([points],self.resolution).astype('float32') #转成voxel

            #if filedir.endswith('.h5'): coords = read_h5_geo(filedir)
            #if filedir.endswith('.ply'): coords = read_ply_ascii_geo(filedir)
            #feats = np.expand_dims(np.ones(coords.shape[0]), 1).astype('int')
            # cache
            self.cache[idx] = points
            cache_percent = int((len(self.cache) / len(self)) * 100)
            if cache_percent > 0 and cache_percent % 10 == 0 and cache_percent != self.last_cache_percent:
                self.last_cache_percent = cache_percent
        #feats = feats.astype("float32")

        return points

#已改
def make_data_loader(dataset, batch_size=32, shuffle=True, num_workers=6, repeat=False, 
                    collate_fn=collate_pointcloud_fn):
    args = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'collate_fn': collate_fn,
        'pin_memory': True, #不适用虚拟内存，即硬盘
        'drop_last': False #最后一个batch会比较小
    }
    if repeat:
        args['sampler'] = InfSampler(dataset, shuffle)
    else:
        args['shuffle'] = shuffle
    loader = torch.utils.data.DataLoader(dataset, **args)

    return loader

#以下测试过
if __name__ == "__main__":
    filedirs = np.array(glob.glob('/userhome/pcc_geo_cnn_v2/pcc_geo_cnn_v2-master/ModelNet40_200_pc512_oct3_4k/'+'**/*.ply', recursive=True))
    files_cat = np.array([os.path.split(os.path.split(x)[0])[1] for x in filedirs])
    files_train = filedirs[files_cat == 'train']
    files_test = filedirs[files_cat == 'test']
    print(files_train[0])
    test_dataset = PCDataset(files_test[:100])
    test_dataloader = make_data_loader(dataset=test_dataset, batch_size=32, shuffle=True, num_workers=2, repeat=False,
                                        collate_fn=collate_pointcloud_fn)
    for idx, points in enumerate(tqdm(test_dataloader)):
        if idx < 1:
            # print("="*20, "check dataset", "="*20, "\npoints:\n", points, "\n")
            print("points[0].shape:",points[0].shape) #points[0].shape: (1, 64, 64, 64)
            print("len(points):",len(points)) #len(points): 32
            x_cpu = torch.from_numpy(np.array(points))
            print("x_cpu.shape:",x_cpu.shape) #x_cpu.shape: torch.Size([32, 1, 64, 64, 64])
            break

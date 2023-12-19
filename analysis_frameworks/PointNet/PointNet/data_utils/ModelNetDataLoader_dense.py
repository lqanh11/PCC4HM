'''
@author: Xu Yan
@file: ModelNet.py
@time: 2021/3/19 15:51
'''
import os
import numpy as np
import warnings
import pickle

from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def write_ply_ascii_geo(filedir, coords):
    if os.path.exists(filedir): os.system('rm '+filedir)
    f = open(filedir,'a+')
    f.writelines(['ply\n','format ascii 1.0\n'])
    f.write('element vertex '+str(coords.shape[0])+'\n')
    f.writelines(['property float x\n',
                  'property float y\n',
                  'property float z\n',
                  'property float nx\n',
                  'property float ny\n',
                  'property float nz\n'])
    f.write('end_header\n')
    coords = coords
    for p in coords:
        f.writelines(f'{int(p[0])} {int(p[1])} {int(p[2])} {p[3]} {p[4]} {p[5]}\n')
    f.close() 

    return

def pc_normalize(pc):
    pc = pc.astype(np.float32)
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader_dense(Dataset):
    def __init__(self, root, split_path, args, split='train', process_data=False):
        self.root = root
        self.npoints = args.num_point
        self.process_data = process_data
        self.uniform = args.use_uniform_sample
        self.use_normals = args.use_normals
        self.num_category = args.num_category
        self.resolution = 2 ** 7 - 1
        self.split_path = split_path

        if self.num_category == 10:
            self.catfile = os.path.join(self.split_path, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.split_path, 'modelnet40_shape_names.txt')

        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.split_path, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.split_path, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.split_path, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.split_path, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))

        if self.uniform:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
            self.save_dense = os.path.join(root, 'modelnet%d_%s_%d_dense.dat' % (self.num_category, split, self.resolution+1))
        else:
            self.save_path = os.path.join(root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))
            self.save_dense = os.path.join(root, 'modelnet%d_%s_%d_dense.dat' % (self.num_category, split, self.resolution+1))

        ply_folder_dir = f'/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet40/pc_normal_resample_format_ply/{self.resolution+1}_dense_with_normal/'
        os.makedirs(ply_folder_dir, exist_ok=True)

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                
                self.list_of_file_name = [None] * len(self.datapath)
                self.list_of_dense_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)
                

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    fn = self.datapath[index]
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
                    
                    self.list_of_file_name[index] = fn

                    ## coordinate_and_normal
                    points_normal = point_set                    
                    ## convert coordinates to interger for PCGCv2
                    xyz = point_set[:,0:3]
                    # coordinates normalize to fixed resolution.
                    for axis_index in range(np.shape(xyz)[1]):
                        xyz[:,axis_index] = xyz[:,axis_index] - np.min(xyz[:,axis_index])
                        xyz[:,axis_index] = xyz[:,axis_index] / np.max(xyz[:,axis_index])
                        xyz[:,axis_index] = xyz[:,axis_index] * (self.resolution)
                    # quantizate to integers. 
                    xyz = np.round(xyz).astype('int')
                    points_normal[:,0:3] = xyz
                    # keep unique
                    xyz, indices = np.unique(xyz, axis=0, return_index=True)

                    points_normal = points_normal[indices, :]
                    ## save_ply
                    shape_name = fn[1].split('/')[-2]
                    txt_filename = fn[1].split('/')[-1]
                    ply_shapename_dir = os.path.join(ply_folder_dir, shape_name)
                    os.makedirs(ply_shapename_dir, exist_ok=True)
                    ply_file_dir = os.path.join(ply_shapename_dir, txt_filename.replace('.txt', '.ply'))
                    write_ply_ascii_geo(ply_file_dir, points_normal)
                    # print('save_ply: ', ply_file_dir)



                    point_set = xyz                    
                    self.list_of_dense_points[index] = xyz
                    
#                     print(fn, 'number of points after preprocessing for dense: ', xyz.shape)


                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
                with open(self.save_dense, 'wb') as f:
                    pickle.dump([self.list_of_file_name, self.list_of_dense_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)

    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        
        point_set = pc_normalize(point_set)
#         if not self.use_normals:
#             point_set = point_set[:, 0:3]
            
      
        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)

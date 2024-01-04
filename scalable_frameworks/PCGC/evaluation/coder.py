import sys
sys.path.append('../')

import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo, write_ply_data_with_normals

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

import h5py
import open3d as o3d

from pcc_model_scalable import PCCModel_Scalable_ForBest


class CoordinateCoder():
    """encode/decode coordinates using gpcc
    """
    def __init__(self, filename):
        self.filename = filename
        self.o_ply_filename = filename + '_o.ply'
        self.d_ply_filename = filename + '_d.ply'

    def encode(self, coords, postfix=''):
        coords = coords.numpy().astype('int')
        write_ply_ascii_geo(filedir=self.o_ply_filename, coords=coords)
        gpcc_encode(self.o_ply_filename, self.filename+postfix+'_C.bin')
        # os.system('rm '+self.ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.d_ply_filename)
        coords = read_ply_ascii_geo(self.d_ply_filename)
        # os.system('rm '+self.ply_filename)
        
        return coords
    
class BaseCoder():
    """encode/decode base branch
    """
    def __init__(self, filename, model):
        self.filename = filename
        self.model = model
        self.coordinate_coder = CoordinateCoder(filename)

    def encode(self, y, postfix=''):
        y_F_q, _ = self.model.entropy_bottleneck(y.F.cpu(), quantize_mode="symbols")
        y_q = ME.SparseTensor(
            features=y_F_q, 
            coordinate_map_key=y.coordinate_map_key, 
            coordinate_manager=y.coordinate_manager, 
            device=y.device)

        z = self.model.adapter(y_q)
        
        z_F_q, _ = self.model.entropy_bottleneck_b(z.F.cpu(), quantize_mode="symbols")
        z_q = ME.SparseTensor(
            features=z_F_q, 
            coordinate_map_key=z.coordinate_map_key, 
            coordinate_manager=z.coordinate_manager, 
            device=z.device)
        
        ## base coordinate
        base_tensor_stride = z_q.tensor_stride[0]
        self.coordinate_coder.encode((z_q.C//base_tensor_stride).detach().cpu()[:,1:], postfix=postfix)
        
        ## base feature
        base_shape = z.F.shape
        base_strings, base_min_v, base_max_v = self.model.entropy_bottleneck_b.compress(z.F.cpu())
        
        with open(self.filename+postfix+'_base_F.bin', 'wb') as fout:
            fout.write(base_strings)
        with open(self.filename+postfix+'_base_H.bin', 'wb') as fout:
            fout.write(np.array(base_shape, dtype=np.int32).tobytes())
            fout.write(np.array(base_tensor_stride, dtype=np.int8).tobytes())
            fout.write(np.array(len(base_min_v), dtype=np.int8).tobytes())
            fout.write(np.array(base_min_v, dtype=np.float32).tobytes())
            fout.write(np.array(base_max_v, dtype=np.float32).tobytes())
  
        return y_q, z_q

    def decode(self, postfix=''):
        
        ## base coordinate
        z_C = self.coordinate_coder.decode(postfix=postfix)
        z_C = torch.cat((torch.zeros((len(z_C),1)).int(), torch.tensor(z_C).int()), dim=-1)
        indices_sort = np.argsort(array2vector(z_C, z_C.max()+1))
        z_C = z_C[indices_sort]
        
        ## base feature
        with open(self.filename+postfix+'_base_F.bin', 'rb') as fin:
            base_strings = fin.read()
        with open(self.filename+postfix+'_base_H.bin', 'rb') as fin:
            base_shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            base_tensor_stride = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            len_base_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            base_min_v = np.frombuffer(fin.read(4*len_base_min_v), dtype=np.float32)[0]
            base_max_v = np.frombuffer(fin.read(4*len_base_min_v), dtype=np.float32)[0]
        z_feats = self.model.entropy_bottleneck_b.decompress(base_strings, base_min_v, base_max_v, base_shape, channels=base_shape[-1])
        
        z_q = ME.SparseTensor(features=z_feats, coordinates=z_C*base_tensor_stride,
                            tensor_stride=base_tensor_stride, device=device)

        return z_q

class Coder():
    def __init__(self, model, filename):
        self.model = model 
        self.filename = filename

        self.base_coder = BaseCoder(self.filename, model)

    @torch.no_grad()
    def encode(self, x, postfix=''):
        # Encoder
        y_list = self.model.encoder(x)
        y = sort_spare_tensor(y_list[0])
        num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
        with open(self.filename+postfix+'_num_points.bin', 'wb') as f:
            f.write(np.array(num_points, dtype=np.int32).tobytes())

        ## base branch
        y_q, z_q_encode_side = self.base_coder.encode(y, postfix=postfix)

        ## enhancement brach
        y_b = self.model.transpose_adapter(z_q_encode_side, [], [], False)

        y_r_features =  y_q.F - y_b.F

        y_r = ME.SparseTensor(
            features=y_r_features, 
            coordinate_map_key=y_q.coordinate_map_key, 
            coordinate_manager=y_q.coordinate_manager, 
            device=y_q.device)
        
        z_r = self.model.analysis_residual(y_r)

        enhanment_shape = z_r.F.shape
        enhanment_strings, enhanment_min_v, enhanment_max_v = self.model.entropy_bottleneck_e.compress(z_r.F.cpu())
        
        with open(self.filename+postfix+'_enhanment_F.bin', 'wb') as fout:
            fout.write(enhanment_strings)
        with open(self.filename+postfix+'_enhanment_H.bin', 'wb') as fout:
            fout.write(np.array(enhanment_shape, dtype=np.int32).tobytes())
            fout.write(np.array(len(enhanment_min_v), dtype=np.int8).tobytes())
            fout.write(np.array(enhanment_min_v, dtype=np.float32).tobytes())
            fout.write(np.array(enhanment_max_v, dtype=np.float32).tobytes())
                
        return 

    @torch.no_grad()
    def decode(self, rho=1, postfix=''):

        ## base branch
        z_q_decode_side = self.base_coder.decode(postfix=postfix)

        # classification
        logits = self.model.classifier(z_q_decode_side)

        # reconstruction
        y_b = self.model.transpose_adapter(z_q_decode_side, [], [], False)

        with open(self.filename+postfix+'_enhanment_F.bin', 'rb') as fin:
            enhanment_strings = fin.read()
        with open(self.filename+postfix+'_enhanment_H.bin', 'rb') as fin:
            enhanment_shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
            len_enhanment_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
            enhanment_min_v = np.frombuffer(fin.read(4*len_enhanment_min_v), dtype=np.float32)[0]
            enhanment_max_v = np.frombuffer(fin.read(4*len_enhanment_min_v), dtype=np.float32)[0]
        z_feats = self.model.entropy_bottleneck_e.decompress(enhanment_strings, enhanment_min_v, enhanment_max_v, enhanment_shape, channels=enhanment_shape[-1])
        
        z_r_q = ME.SparseTensor(features=z_feats, 
                                coordinate_map_key=y_b.coordinate_map_key, 
                                coordinate_manager=y_b.coordinate_manager, 
                                device=y_b.device)
        y_r_hat = self.model.systhesis_residual(z_r_q, [], [], False)
        
        y_scalable_features = y_r_hat.F + y_b.F
        
        y_scalable = ME.SparseTensor(
            features=y_scalable_features, 
            coordinate_map_key=y_r_hat.coordinate_map_key, 
            coordinate_manager=y_r_hat.coordinate_manager, 
            device=y_r_hat.device)

        # decode label
        with open(self.filename+postfix+'_num_points.bin', 'rb') as fin:
            num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            num_points[-1] = int(rho * num_points[-1])# update
            num_points = [[num] for num in num_points]
        # decode
        _, out = self.model.decoder(y_scalable, nums_list=num_points, ground_truth_list=[None]*3, training=False)

        return out

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0107.h5')
    parser.add_argument("--outdir", default='./output/ModelNet/scalable')
    parser.add_argument("--ckptdir", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/20231231_encFIXa10_baseFIXc4_enhaTRAINc8b0_Quantize_MSE_resolution128_alpha1.0_010/ckpts/epoch_9.pth') 
    parser.add_argument("--res", type=int, default=128, help='resolution')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    
    args = parser.parse_args()

    output_resolution = os.path.join(args.outdir, f'{args.res}')
    if not os.path.exists(output_resolution): os.makedirs(output_resolution)
    
    ## save PC with .ply format from .h5 file 
    save_original_path = os.path.join(output_resolution, 'original')
    os.makedirs(save_original_path, exist_ok=True)

    pc_filedir = os.path.join(save_original_path, os.path.split(args.filedir)[-1].split('.')[0] + '.ply')

    # read h5 file
    h5_name = args.filedir
    with h5py.File(h5_name) as f:
        dense_xyz = f[f"points_{args.res}"][:].astype("int")
        label = f["class"][:].astype("int64")
    # estimate normals
    pcd = o3d.geometry.PointCloud()
    pcd.points=o3d.utility.Vector3dVector(dense_xyz)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10, max_nn=20))
    write_ply_data_with_normals(pc_filedir, np.asarray(pcd.points), np.asarray(pcd.normals))

    # load data
    start_time = time.time()
    x = load_sparse_tensor(pc_filedir, device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    outdir = args.outdir
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.split(pc_filedir)[-1].split('.')[0]
    filename = os.path.join(outdir, filename)
    print(filename)

    # model
    print('='*10, 'Test', '='*10)
    model = PCCModel_Scalable_ForBest(
        encoder_channels = [1,16,32,64,32,8],
        decoder_channels = [8,64,32,16],
        adapter_channels = [8,4],
        t_adapter_channels = [4,8],

        num_cls = 10,
        embedding_channel_cls = 1024,

        analysis_channels = [8,32,8],
        synthesis_channels = [8,32,8],
        layer_blocks = 3
    ).to(device)
    
    assert os.path.exists(args.ckptdir)
    ckpt = torch.load(args.ckptdir)
    model.load_state_dict(ckpt['model'])
    print('load checkpoint from \t', args.ckptdir)

    model.eval()

    # coder
    coder = Coder(model=model, filename=filename)

    # down-scale
    if args.scaling_factor!=1: 
        x_in = scale_sparse_tensor(x, factor=args.scaling_factor)
    else: 
        x_in = x

    # encode
    start_time = time.time()
    coder.encode(x_in)
    
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    x_dec = coder.decode(rho=args.rho)
    print('Dec Time:\t', round(time.time() - start_time, 3), 's')

    # up-scale
    if args.scaling_factor!=1: 
        x_dec = scale_sparse_tensor(x_dec, factor=1.0/args.scaling_factor)

    # bitrate
    bits = np.array([os.path.getsize(filename + postfix)*8 \
                            for postfix in ['_C.bin', 
                                            '_base_F.bin',
                                            '_base_H.bin', 
                                            '_enhanment_F.bin',
                                            '_enhanment_H.bin', 
                                            '_num_points.bin']])
    bpps = (bits/len(x)).round(3)
    print('bits:\t', bits, '\nbpps:\t', bpps)
    print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

    start_time = time.time()
    pc_error_metrics = pc_error(args.filedir, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])

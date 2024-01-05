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

from pcc_model_scalable import PCCModel_Classification_Adapter_KeepCoords

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
        os.system('rm '+self.o_ply_filename)
        
        return 

    def decode(self, postfix=''):
        gpcc_decode(self.filename+postfix+'_C.bin', self.d_ply_filename)
        coords = read_ply_ascii_geo(self.d_ply_filename)
        os.system('rm '+self.d_ply_filename)
        
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
        
        y_b_hat = model.latentspacetransform(z_q)
        

        return y_b_hat

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
        ## base branch
        y_q, z_q_encode_side = self.base_coder.encode(y, postfix=postfix)
                
        return y_q, z_q_encode_side

    @torch.no_grad()
    def decode(self, postfix=''):
        ## base branch
        y_b_decode_side = self.base_coder.decode(postfix=postfix)
        # classification
        logits = self.model.classifier(y_b_decode_side)

        return logits

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--list_txt", 
                        default="/media/avitech/Data/quocanhle/PointCloud/source_code/scalable_frameworks/PCGC/evaluation/test_ModelNetNet_all.txt")
    parser.add_argument("--outdir", default='./output/ModelNet/scalable')
    
    parser.add_argument("--init_ckpt_original", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/2024-01-04_22-17_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000/src/ckpt_original/epoch_10.pth')
    parser.add_argument("--init_ckpt_base", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/2024-01-04_22-17_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000/src/ckpt_base/epoch_195.pth')
    
    parser.add_argument("--res", type=int, default=128, help='resolution')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    
    args = parser.parse_args()

    # model
    print('='*10, 'Test', '='*10)
    model = PCCModel_Classification_Adapter_KeepCoords().to(device)
    model_dict = model.state_dict() 
    processed_dict = {}

    # load pre-trained model
    if args.init_ckpt_original != '':
        state_dict = torch.load(args.init_ckpt_original)
        model_compression_dict_original = state_dict["model"]
        
        for k in model_dict.keys(): 
            decomposed_key = k.split(".")
            if("encoder" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_original[pretrained_key] 
            if("decoder" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_original[pretrained_key]
            if("entropy_bottleneck" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_original[pretrained_key]
    # load base branch
    if args.init_ckpt_base != '':
        state_dict = torch.load(args.init_ckpt_base)
        model_compression_dict_base = state_dict["model"]
        for k in model_dict.keys(): 
            decomposed_key = k.split(".")
            if("entropy_bottleneck_b" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("adapter" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("latentspacetransform" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
            if("classifier" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_base[pretrained_key]
    for k in processed_dict.keys(): 
        # print('Load weight: ', k)
        model_dict[k] = processed_dict[k]

    model.load_state_dict(model_dict)

    model.eval()

    output_resolution = os.path.join(args.outdir, f'{args.res}')
    if not os.path.exists(output_resolution): os.makedirs(output_resolution)
    
    ## save PC with .ply format from .h5 file 
    save_original_path = os.path.join(output_resolution, 'original')
    os.makedirs(save_original_path, exist_ok=True)
    with open(os.path.join(args.list_txt), 'r') as file:
        lines = file.readlines()
        for idx, line in enumerate(lines[:1]):
            print(line.strip())
            args.filedir = line.strip()

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
            logits = coder.decode()

            preds_class = torch.argmax(logits, 1)

            print('Dec Time:\t', round(time.time() - start_time, 3), 's')

            # bitrate
            bits = np.array([os.path.getsize(filename + postfix)*8 \
                                    for postfix in ['_C.bin', 
                                                    '_base_F.bin',
                                                    '_base_H.bin', 
                                                    ]])
            bpps = (bits/len(x)).round(3)
            print('bits:\t', bits, '\nbpps:\t', bpps)
            print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))
            print(preds_class)

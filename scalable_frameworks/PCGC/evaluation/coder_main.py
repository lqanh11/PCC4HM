import sys
sys.path.append('../')

import os, time
import numpy as np
import torch
import MinkowskiEngine as ME
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo, write_ply_data_with_normals
from loss import get_cls_metrics

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

import h5py
import open3d as o3d

from pcc_model_scalable import PCCModel_Scalable_ForBest_KeepCoords


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

if __name__ == '__main__':
    sys.argv = ['notebook_name']
    
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0117.h5')
    parser.add_argument("--outdir", default='./output/ModelNet/scalable')
    
    parser.add_argument("--init_ckpt_original", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/2024-01-04_22-17_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000/src/ckpt_original/epoch_10.pth')
    parser.add_argument("--init_ckpt_base", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/2024-01-04_22-17_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000/src/ckpt_base/epoch_195.pth')
    parser.add_argument("--init_ckpt", default='/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/2024-01-04_22-17_encFIXa025_baseFIXc4_enhaTRAINc8b3RB_Quantize_MSE_KeepCoords_resolution128_alpha1.0_000/best_model/epoch_5.pth')
    
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
    model = PCCModel_Scalable_ForBest_KeepCoords(
        encoder_channels = [1,16,32,64,32,8],
        decoder_channels = [8,64,32,16],
        adapter_channels = [8,4,4],
        t_adapter_channels = [4,4,8],

        num_cls = 10,
        embedding_channel_cls = 1024,

        analysis_channels = [8,32,8],
        synthesis_channels = [8,32,8],
        layer_blocks = 3
    ).to(device)
    
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

    # load enhanment brach
    if args.init_ckpt != '':
        state_dict = torch.load(args.init_ckpt)
        model_compression_dict_enhance = state_dict["model"]
        for k in model_dict.keys(): 
            decomposed_key = k.split(".")

            if("entropy_bottleneck_e" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("transpose_adapter" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("analysis_residual" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
            if("systhesis_residual" in decomposed_key):
                pretrained_key = ".".join(decomposed_key[:])
                processed_dict[k] = model_compression_dict_enhance[pretrained_key]
    
    for k in processed_dict.keys(): 
        # print('Load weight: ', k)
        model_dict[k] = processed_dict[k]

    model.load_state_dict(model_dict)

    model.eval()

    # down-scale
    if args.scaling_factor!=1: 
        x_in = scale_sparse_tensor(x, factor=args.scaling_factor)
    else: 
        x_in = x 
    

    # encode
    start_time = time.time()
    coordinate_coder = CoordinateCoder(filename)


    # Encoder
    y_list = model.encoder(x)
    y = sort_spare_tensor(y_list[0])
    num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
    with open(filename+'_num_points.bin', 'wb') as f:
        f.write(np.array(num_points, dtype=np.int32).tobytes())

    ## base branch
    y_F_q, _ = model.entropy_bottleneck(y.F.cpu(), quantize_mode="symbols")
    y_q = ME.SparseTensor(
        features=y_F_q, 
        coordinate_map_key=y.coordinate_map_key, 
        coordinate_manager=y.coordinate_manager, 
        device=y.device)

    z = model.adapter(y_q)
    
    z_F_q, _ = model.entropy_bottleneck_b(z.F.cpu(), quantize_mode="symbols")
    
    ## base coordinate
    base_tensor_stride = z.tensor_stride[0]

    indices_sort = np.argsort(array2vector(z.C, z.C.max()+1))
    z_C = z.C[indices_sort]
    z_F = z_F_q[indices_sort]

    z_q_encode_side = ME.SparseTensor(features=z_F, coordinates=z_C,
                        tensor_stride=base_tensor_stride, device=device)

    coordinate_coder.encode((z_C//base_tensor_stride).detach().cpu()[:,1:], postfix='')
    
    ## base feature
    base_shape = z.F.shape
    base_strings, base_min_v, base_max_v = model.entropy_bottleneck_b.compress(z.F.cpu())
    
    with open(filename+'_base_F.bin', 'wb') as fout:
        fout.write(base_strings)
    with open(filename+'_base_H.bin', 'wb') as fout:
        fout.write(np.array(base_shape, dtype=np.int32).tobytes())
        fout.write(np.array(base_tensor_stride, dtype=np.int8).tobytes())
        fout.write(np.array(len(base_min_v), dtype=np.int8).tobytes())
        fout.write(np.array(base_min_v, dtype=np.float32).tobytes())
        fout.write(np.array(base_max_v, dtype=np.float32).tobytes())    

    nums_list_for_e = [[len(C) for C in y.decomposed_coordinates] \
            for y in [y_q]]
    with open(filename+'_num_points_enhanment.bin', 'wb') as f:
        f.write(np.array(nums_list_for_e, dtype=np.int32).tobytes())

    ## enhancement brach
    y_b_encoder_side = model.transpose_adapter(z_q_encode_side)

    y_r_features =  y_q.F - y_b_encoder_side.F

    y_r = ME.SparseTensor(
        features=y_r_features, 
        coordinate_map_key=y_b_encoder_side.coordinate_map_key, 
        coordinate_manager=y_b_encoder_side.coordinate_manager, 
        device=y_q.device)
    
    z_r = model.analysis_residual(y_r)

    enhanment_shape = z_r.F.shape
    enhanment_strings, enhanment_min_v, enhanment_max_v = model.entropy_bottleneck_e.compress(z_r.F.cpu())
    
    with open(filename+'_enhanment_F.bin', 'wb') as fout:
        fout.write(enhanment_strings)
    with open(filename+'_enhanment_H.bin', 'wb') as fout:
        fout.write(np.array(enhanment_shape, dtype=np.int32).tobytes())
        fout.write(np.array(len(enhanment_min_v), dtype=np.int8).tobytes())
        fout.write(np.array(enhanment_min_v, dtype=np.float32).tobytes())
        fout.write(np.array(enhanment_max_v, dtype=np.float32).tobytes())
    
    print('Enc Time:\t', round(time.time() - start_time, 3), 's')

    # decode
    start_time = time.time()
    
    ## base branch
    ## base coordinate
    z_C = coordinate_coder.decode(postfix='')
    z_C = torch.cat((torch.zeros((len(z_C),1)).int(), torch.tensor(z_C).int()), dim=-1)
    indices_sort = np.argsort(array2vector(z_C, z_C.max()+1))
    z_C = z_C[indices_sort]
    
    ## base feature
    with open(filename+'_base_F.bin', 'rb') as fin:
        base_strings = fin.read()
    with open(filename+'_base_H.bin', 'rb') as fin:
        base_shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
        base_tensor_stride = np.frombuffer(fin.read(1), dtype=np.int8)[0]
        len_base_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
        base_min_v = np.frombuffer(fin.read(4*len_base_min_v), dtype=np.float32)[0]
        base_max_v = np.frombuffer(fin.read(4*len_base_min_v), dtype=np.float32)[0]
    z_feats = model.entropy_bottleneck_b.decompress(base_strings, base_min_v, base_max_v, base_shape, channels=base_shape[-1])
    
    z_q_decode_side = ME.SparseTensor(features=z_feats, coordinates=z_C*base_tensor_stride,
                        tensor_stride=base_tensor_stride, device=device)

    # classification
    logits = model.classifier(model.latentspacetransform(z_q_decode_side))

    # reconstruction
    with open(filename+'_num_points_enhanment.bin', 'rb') as fin:
        nums_list_for_e = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
        nums_list_for_e[-1] = int(args.rho * nums_list_for_e[-1])# update
        nums_list_for_e = [[num] for num in nums_list_for_e]

    y_b_decode_side = model.transpose_adapter(z_q_decode_side)

    with open(filename+'_enhanment_F.bin', 'rb') as fin:
        enhanment_strings = fin.read()
    with open(filename+'_enhanment_H.bin', 'rb') as fin:
        enhanment_shape = np.frombuffer(fin.read(4*2), dtype=np.int32)
        len_enhanment_min_v = np.frombuffer(fin.read(1), dtype=np.int8)[0]
        enhanment_min_v = np.frombuffer(fin.read(4*len_enhanment_min_v), dtype=np.float32)[0]
        enhanment_max_v = np.frombuffer(fin.read(4*len_enhanment_min_v), dtype=np.float32)[0]
    z_feats = model.entropy_bottleneck_e.decompress(enhanment_strings, enhanment_min_v, enhanment_max_v, enhanment_shape, channels=enhanment_shape[-1])
    
    z_r_q = ME.SparseTensor(features=z_feats, 
                            coordinate_map_key=y_b_decode_side.coordinate_map_key, 
                            coordinate_manager=y_b_decode_side.coordinate_manager, 
                            device=y_b_decode_side.device)
    y_r_hat = model.systhesis_residual(z_r_q)
    
    y_scalable_features = y_r_hat.F + y_b_decode_side.F
    
    y_scalable = ME.SparseTensor(
        features=y_scalable_features, 
        coordinate_map_key=y_r_hat.coordinate_map_key, 
        coordinate_manager=y_r_hat.coordinate_manager, 
        device=y_r_hat.device)

    # decode label
    with open(filename+'_num_points.bin', 'rb') as fin:
        num_points = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
        num_points[-1] = int(args.rho * num_points[-1])# update
        num_points = [[num] for num in num_points]
    # decode
    _, out = model.decoder(y_scalable, nums_list=num_points, ground_truth_list=[None]*3, training=False)


    print('Dec Time:\t', round(time.time() - start_time, 3), 's')

    x_dec = out
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

    metrics_calculator = []
    for out_cls, ground_truth in zip([x_dec], [x_in]):
        metrics_calculator.append(get_cls_metrics(out_cls.C, ground_truth.C))

    print(metrics_calculator)

    # distortion
    start_time = time.time()
    write_ply_ascii_geo(filename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
    print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

    start_time = time.time()
    pc_error_metrics = pc_error(pc_filedir, filename+'_dec.ply', res=args.res, show=False)
    print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
    # print('pc_error_metrics:', pc_error_metrics)
    print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
    print('D2 PSNR:\t', pc_error_metrics["mseF,PSNR (p2plane)"][0])

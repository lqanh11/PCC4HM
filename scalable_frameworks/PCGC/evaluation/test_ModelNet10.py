import sys
sys.path.append('../')

import torch
import numpy as np
import os



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--filedir", default='/media/avitech/Data/quocanhle/PointCloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/test/bathtub_0107.h5')
    parser.add_argument("--outdir", default='./output/ModelNet/scalable')
    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=128, help='resolution')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    parser.add_argument('--ckptdir_list',
                    type=str, nargs='*', default=[
                                                    '/media/avitech/Data/quocanhle/PointCloud/logs/PCGC_scalable/logs_ModelNet10/enhanment_brach/20231231_encFIXa10_baseFIXc4_enhaTRAINc8b0_Quantize_MSE_resolution128_alpha1.0_010/ckpts/epoch_9.pth',
                                                    ], help="CKPT list")


    args = parser.parse_args()
   
    output_resolution = os.path.join(args.outdir, f'{args.res}')
    if not os.path.exists(output_resolution): os.makedirs(output_resolution)
    
    ## save PC with .ply format from .h5 file 
    save_original_path = os.path.join(output_resolution, 'original')
    os.makedirs(save_original_path, exist_ok=True)

    pc_filedir = os.path.join(save_original_path, os.path.split(args.filedir)[-1].split('.')[0] + '.ply')

    # ## resume
    # check_csv_name = os.path.join(output_resolution, 'r'+str(len(args.ckptdir_list)), os.path.split(pc_filedir)[-1].split('.')[0] +'.csv')
    # if os.path.exists(check_csv_name):
    #     print(check_csv_name)
    # else:
    from pcc_model_scalable import PCCModel_Scalable_ForBest
    from coder import Coder
    import time
    from data_utils import load_sparse_tensor, scale_sparse_tensor
    from data_utils import write_ply_ascii_geo, write_ply_data_with_normals
    import pandas as pd
    import h5py
    import open3d as o3d


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

    # load input PC
    x = load_sparse_tensor(pc_filedir, device)

    # load model
    # model
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
    ).to(device=device)

    for idx, ckptdir in enumerate(args.ckptdir_list):
        
        # postfix: rate index
        postfix_idx = 'r'+str(idx+1)
        postfix_file = ''

        # initialize output filename
        save_results_path = os.path.join(output_resolution, postfix_idx)
        if not os.path.exists(save_results_path): os.makedirs(save_results_path)
        basename = os.path.split(pc_filedir)[-1].split('.')[0]
        filename = os.path.join(save_results_path, basename)
        print('output filename:\t', filename)

        ## Start
        print('='*10, idx+1, '='*10)
        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load checkpoint from \t', ckptdir)
        model.eval()

        coder = Coder(model=model, filename=filename)
        if args.scaling_factor!=1: 
            x_in = scale_sparse_tensor(x, factor=args.scaling_factor)
        else: 
            x_in = x
        
        # encode
        start_time = time.time()
        _ = coder.encode(x_in, postfix=postfix_file)
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode
        start_time = time.time()
        x_dec = coder.decode(postfix=postfix_file, rho=args.rho)
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)

        # up-scale
        if args.scaling_factor!=1: 
            x_dec = scale_sparse_tensor(x_dec, factor=1.0/args.scaling_factor)

        # bitrate
        bits = np.array([os.path.getsize(filename + postfix_file + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
        
        write_ply_ascii_geo(os.path.join(save_results_path, basename +'_dec.ply'), 
                            x_dec.C.detach().cpu().numpy()[:,1:])
        
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        bpps = (bits/len(x)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))
        
        
        results = {}
        results["num_points(input)"] = len(x)
        results["num_points(output)"] = len(x_dec)
        results["resolution"] = args.res
        results["bits"] = sum(bits).round(3)
        results["bpp"] = sum(bpps).round(3)
        results["bpp(coords)"] = bpps[0]
        results["bpp(feats)"] = bpps[1]
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec

        df_results = pd.DataFrame([results])

        csv_name = os.path.join(save_results_path, basename +'.csv')
        df_results.to_csv(csv_name, index=False)
        print('Wrile results to: \t', csv_name)
        

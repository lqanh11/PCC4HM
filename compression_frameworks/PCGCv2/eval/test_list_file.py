import sys
sys.path.append('../')

import torch
import numpy as np
import os
from pcc_model import PCCModel
from coder import Coder
import time
from data_utils import load_sparse_tensor_h5_all, sort_spare_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo
from pc_error import pc_error
import pandas as pd
from tqdm import tqdm
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(filedir, ckptdir_list, experiment_name_list, outdir_all, resultdir, scaling_factor=1.0, rho=1.0, res=1024, normal=False):
    # load data
    start_time = time.time()
    print(filedir)
    x, coords_original = load_sparse_tensor_h5_all(filedir, resolution=res, device=device)




    print('Loading Time:\t', round(time.time() - start_time, 4), 's')
    # x = sort_spare_tensor(input_data)
    
    # load model
    model = PCCModel().to(device)

    for idx, ckptdir in enumerate(ckptdir_list):
        
        outdir = os.path.join(outdir_all, experiment_name_list[idx])
        if not os.path.exists(outdir): os.makedirs(outdir)

        filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
        # write orginal point cloud
        write_ply_ascii_geo(filedir=filename +'_orginal.ply', coords= coords_original)

        print('output filename:\t', filename)

        print('='*10, idx+1, '='*10)
        # load checkpoints
        print(ckptdir)
        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load checkpoint from \t', ckptdir)
        coder = Coder(model=model, filename=filename)

        # postfix: rate index
        postfix_idx = '_r'+str(idx+1)

        # down-scale
        if scaling_factor!=1: 
            x_in = scale_sparse_tensor(x, factor=scaling_factor)
        else: 
            x_in = x

        # encode
        start_time = time.time()
        _ = coder.encode(x_in, postfix=postfix_idx)
        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode
        start_time = time.time()
        x_dec = coder.decode(postfix=postfix_idx, rho=rho)
        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)

        # up-scale
        if scaling_factor!=1: 
            x_dec = scale_sparse_tensor(x_dec, factor=1.0/scaling_factor)

        # bitrate
        bits = np.array([os.path.getsize(filename + postfix_idx + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])
        bpps = (bits/len(x)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

        # distortion
        start_time = time.time()
        write_ply_ascii_geo(filename+postfix_idx+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')

        start_time = time.time()
        pc_error_metrics = pc_error(filename +'_orginal.ply', filename+postfix_idx+'_dec.ply', 
                                    res=res, normal=normal, show=False)
        
        print(pc_error_metrics)
        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
        # print('D2 PSNR:\t', pc_error_metrics["mseF,PSNR (p2plane)"][0])

        # save results
        results = pc_error_metrics
        results["num_points(input)"] = len(x)
        results["num_points(output)"] = len(x_dec)
        results["resolution"] = res
        results["bits"] = sum(bits).round(3)
        results["bits"] = sum(bits).round(3)
        results["bpp"] = sum(bpps).round(3)
        results["bpp(coords)"] = bpps[0]
        results["bpp(feats)"] = bpps[1]
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec
        if idx == 0:
            all_results = results.copy(deep=True)
        else: 
            all_results = all_results._append(results, ignore_index=True)

    return all_results

        
def get_file_dirs(data_split_path, phase = 'test'):

    test_file_dirs = []

    list_file = os.listdir(os.path.join(data_split_path, phase))
        
    for file_dir in list_file:
        test_file_dirs.append(os.path.join(data_split_path, phase, file_dir))

    print('Test: ', len(test_file_dirs))
    
    return {'Test':test_file_dirs}
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_split_path", default='/mnt/Data/quocanhle/Point_Cloud/dataset/modelnet10/pc_resample_format_h5/all_resolution/')
    parser.add_argument("--outdir", default='/mnt/Data/quocanhle/Point_Cloud/save_results/')
    parser.add_argument("--resultdir", default='/mnt/Data/quocanhle/Point_Cloud/save_results/')

    parser.add_argument("--scaling_factor", type=float, default=1.0, help='scaling_factor')
    parser.add_argument("--res", type=int, default=128, help='resolution')
    parser.add_argument("--rho", type=float, default=1.0, help='the ratio of the number of output points to the number of input points')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)

    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    
    experiment_name_list = [
        'r1_finetune_modelnet10', 
        'r2_finetune_modelnet10', 
        'r3_finetune_modelnet10', 
        'r4_finetune_modelnet10', 
        'r5_finetune_modelnet10', 
        'r6_finetune_modelnet10', 
        'r7_finetune_modelnet10']

    ckptdir_list = [
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.25_000/ckpts/epoch_10.pth',
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_0.5_000/ckpts/epoch_10.pth',
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_1.0_000/ckpts/epoch_10.pth',
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_2.0_000/ckpts/epoch_10.pth',
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_4.0_000/ckpts/epoch_10.pth',
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_6.0_000/ckpts/epoch_10.pth',
        '/mnt/Data/quocanhle/Point_Cloud/pretrained_models/PCGCv2/fineturning_ModelNet10/modelnet_dense_full_reconstruction_with_pretrained_alpha_10.0_000/ckpts/epoch_10.pth',
                    ]
    
    save_results = {}

    for idx_ckpt, ckptdir in enumerate(ckptdir_list):
        experiment_name = experiment_name_list[idx_ckpt]
        # if os.path.exists(os.path.join(args.resultdir, experiment_name + '.txt')):
        #     os.remove(os.path.join(args.resultdir, experiment_name + '.txt'))
        save_results[experiment_name] = open(os.path.join(args.resultdir, experiment_name + '.txt'), 'a+')

    filedirs = get_file_dirs(args.data_split_path)
    for index_file, filedir in tqdm(enumerate(filedirs['Test'])):
        
        if index_file >=285:
            print(index_file)
            all_results = test(filedir, ckptdir_list, experiment_name_list, args.outdir, args.resultdir, scaling_factor=args.scaling_factor, rho=args.rho, res=args.res, normal=False)
            
            for idx_ckpt, ckptdir in enumerate(ckptdir_list):
                experiment_name = experiment_name_list[idx_ckpt]
                

                if index_file == 0:
                    print('Save_header.....')
                    ## save header
                    save_results[experiment_name].write(f'Filename\t')
                    for key in all_results.keys():
                        save_results[experiment_name].write(f'{key.replace(" ", "")}\t')
                    save_results[experiment_name].write(f'\n')

                    ## save result value
                    save_results[experiment_name].write(f'{filedir}\t')   
                    for result_value in pd.to_numeric(all_results.values[idx_ckpt]):
                        save_results[experiment_name].write(f'{result_value}\t')
                    save_results[experiment_name].write(f'\n')    
                else:
                    ## save result value
                    save_results[experiment_name].write(f'{filedir}\t')   
                    for result_value in pd.to_numeric(all_results.values[idx_ckpt]):
                        save_results[experiment_name].write(f'{result_value}\t')
                    save_results[experiment_name].write(f'\n')    

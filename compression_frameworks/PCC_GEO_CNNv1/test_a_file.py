import torch
import numpy as np
import os
from compression_model import PCCModel
import time
from data_utils import write_ply_ascii_geo
from pc_error import pc_error
import pandas as pd
from tqdm import tqdm
from pyntcloud import PyntCloud
from inout_points import points2voxels
import gzip
import pc_io
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TYPE = np.uint16
DTYPE = np.dtype(np.uint16)
SHAPE_LEN = 3
def compress(nn_output):
    x_shape = nn_output['x_shape'] #x_shape: [512 512 512]
    y_shape = nn_output['y_shape'] #y_shape: [64 64 64]
    string = nn_output['string']
    x_shape_b = np.array(x_shape, dtype=TYPE).tobytes()
    y_shape_b = np.array(y_shape, dtype=TYPE).tobytes()
    representation = x_shape_b + y_shape_b + string

    return representation

def load_compressed_file(file):
    with gzip.open(file, "rb") as f:
        x_shape = np.frombuffer(f.read(DTYPE.itemsize * SHAPE_LEN), dtype=TYPE)
        y_shape = np.frombuffer(f.read(DTYPE.itemsize * SHAPE_LEN), dtype=TYPE)
        string = f.read()

        return x_shape, y_shape, string

def quantize_tensor(x):
    # x = tf.clip_by_value(x, 0, 1)
    x = torch.clamp(x, min=0, max=1)

    x = torch.round(x)
    # x = tf.cast(x, tf.uint8)
    x = x.int()
    return x.cpu().numpy()

def test(filedir, ckptdir_list, outdir, normal=False):
    # load data
    start_time = time.time()
    print(filedir)
    pc = PyntCloud.from_file(filedir)
    points = pc.points[['x','y','z']].values.astype('int32')
    max = points.max()
    i = 1
    while(True):
        if 2 ** i > max:
            resolution = 2 ** i
            break
        else:
            i += 1

    print(resolution)
    x_np = points2voxels([points],resolution).astype('float32')
    x_cpu = torch.from_numpy(x_np).permute(0,4,1,2,3)
    x = x_cpu.to(device)
    print('Loading Time:\t', round(time.time() - start_time, 4), 's')

    model = PCCModel(num_filters=32)
    model = model.to(device)
    # output filename
    if not os.path.exists(outdir): os.makedirs(outdir)
    filename = os.path.join(outdir, os.path.split(filedir)[-1].split('.')[0])
    print('output filename:\t', filename)

    for idx, ckptdir in enumerate(ckptdir_list):
        print('='*10, idx+1, '='*10)
        # load checkpoints
        print(ckptdir)
        assert os.path.exists(ckptdir)
        ckpt = torch.load(ckptdir)
        model.load_state_dict(ckpt['model'])
        print('load checkpoint from \t', ckptdir)
        
        # postfix: rate index
        postfix_idx = '_r'+str(idx+1)

        # encode
        start_time = time.time()

        y = model.analysis_transform(x)
        y_strings = model.entropy_bottleneck.compress(y)
        x_shape = x_cpu.shape[2:]
        y_shape = y.shape[2:]

        ret = {'string':y_strings, 'x_shape':x_shape, 'y_shape':y_shape}

        output_file = filename + f'{postfix_idx}.bin'
        with gzip.open(output_file, "wb") as f:
            representation = compress(ret)
            f.write(representation)

        print('Enc Time:\t', round(time.time() - start_time, 3), 's')
        time_enc = round(time.time() - start_time, 3)

        # decode       
        start_time = time.time()
        
        compressed_data = load_compressed_file(output_file)
        x_shape = compressed_data[0]
        y_shape = compressed_data[1]
        y_shape = [1,32] + [int(s) for s in y_shape]

        compressed_strings = compressed_data[2]
        y_hat = model.entropy_bottleneck.decompress(compressed_strings, y_shape)
        x_hat = model.synthesis_transform(y_hat.to(device))
        x_hat = x_hat[:, :, :x_shape[0], :x_shape[1], :x_shape[2]]
        x_hat_quant = quantize_tensor(x_hat)

        pa = np.argwhere(x_hat_quant[0][0]).astype('float32') #返回非0的数组元组的索引

        print('Dec Time:\t', round(time.time() - start_time, 3), 's')
        time_dec = round(time.time() - start_time, 3)

        # bitrate
        bits = np.array([os.path.getsize(output_file)])
        bpps = (bits/len(points)).round(3)
        print('bits:\t', sum(bits), '\nbpps:\t',  sum(bpps).round(3))

        # distortion
        start_time = time.time()
        reconstruction_file = filename + f'{postfix_idx}_dec.ply'
        x_dec = pa
        write_ply_ascii_geo(reconstruction_file, x_dec)
        # pc_io.write_df(reconstruction_file, pc_io.pa_to_df(pa))
        print('Write PC Time:\t', round(time.time() - start_time, 3), 's')
        
        

        start_time = time.time()
        pc_error_metrics = pc_error(filedir, 
                                    reconstruction_file, 
                                    res=resolution, 
                                    normal=normal, 
                                    show=False)
        
        print(pc_error_metrics)

        print('PC Error Metric Time:\t', round(time.time() - start_time, 3), 's')
        print('D1 PSNR:\t', pc_error_metrics["mseF,PSNR (p2point)"][0])
        print('D2 PSNR:\t', pc_error_metrics["mseF,PSNR (p2plane)"][0])

        # save results
        results = pc_error_metrics
        results["num_points(input)"] = len(x)
        results["num_points(output)"] = len(x_dec)
        results["resolution"] = resolution
        results["bits"] = sum(bits).round(3)
        results["bpp"] = sum(bpps).round(3)
        results["bpp(coords)"] = sum(bpps)
        results["bpp(coords)"] = sum(bpps)
        results["time(enc)"] = time_enc
        results["time(dec)"] = time_dec
        if idx == 0:
            all_results = results.copy(deep=True)
        else: 
            all_results = all_results._append(results, ignore_index=True)

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--filedir", default='/media/avitech/CODE/quocanhle/Point_Cloud_Compression/dataset/testdata/MVUB/sarah_vox9_frame0023.ply')
    parser.add_argument("--outdir", default='./output')
    parser.add_argument("--resultdir", default='./results')
    args = parser.parse_args()

    if not os.path.exists(args.outdir): os.makedirs(args.outdir)
    if not os.path.exists(args.resultdir): os.makedirs(args.resultdir)
    
    ckptdir_list = [
                    './models/1e-05/ckpts/epoch_18_22517.pth',
                    './models/5e-05/ckpts/epoch_19_23448.pth',
                    './models/0.0001/ckpts/epoch_9_11998.pth',
                    './models/0.0005/ckpts/epoch_19_23488.pth',
                    './models/0.001/ckpts/epoch_18_22177.pth',
                    './models/0.005/ckpts/epoch_7_9396.pth',                                     
                    ]
    all_results = test(args.filedir, ckptdir_list, args.outdir, normal=True)

    # save to csv
    csv_name = os.path.join(args.resultdir, os.path.split(args.filedir)[-1].split('.')[0]+'.csv')
    all_results.to_csv(csv_name, index=False)
    print('Wrile results to: \t', csv_name)

    # plot RD-curve
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2point)"][:]), 
            label="D1", marker='x', color='red')
    plt.plot(np.array(all_results["bpp"][:]), np.array(all_results["mseF,PSNR (p2plane)"][:]), 
            label="D2", marker='x', color='blue')
    filename = os.path.split(args.filedir)[-1][:-4]
    plt.title(filename)
    plt.xlabel('bpp')
    plt.ylabel('PSNR')
    plt.grid(ls='-.')
    plt.legend(loc='lower right')
    fig.savefig(os.path.join(args.resultdir, filename+'.jpg'))


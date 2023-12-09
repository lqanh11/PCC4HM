import os
from re import I
import numpy as np
import torch
import argparse
from pyntcloud import PyntCloud
from inout_points import points2voxels
from compression_model import PCCModel
import gzip

np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################################################################
### Script
################################################################################
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compress.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_file',type=str, default='28_airplane_0270.ply', # /userhome/dataset/paper_test/longdress_vox10_1300.ply
        help='Input directory.')
    parser.add_argument(
        '--output_dir',type=str, default='output/',
        help='Output directory.')
    parser.add_argument(
        "--init_ckpt", type=str, default='models/epoch_20_6219.pth', dest="init_ckpt",
        help='initial checkpoint directory.')
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')

    args = parser.parse_args()

    pc = PyntCloud.from_file(args.input_file)
    points = pc.points[['x','y','z']].values.astype('int32')
    max = points.max()
    i = 1
    while(True):
        if 2 ** i > max:
            resolution = 2 ** i
            break
        else:
            i += 1
    x_np = points2voxels([points],resolution).astype('float32') #转成voxel

    x_cpu = torch.from_numpy(x_np).permute(0,4,1,2,3) #转成torch的tensor格式 (8, 64, 64, 64, 1)->(8, 1, 64, 64, 64)
    x = x_cpu.to(device) #转到GPU
    # forward
    model = PCCModel(num_filters=args.num_filters)
    model = model.to(device)
    ckpt = torch.load(args.init_ckpt)
    model.load_state_dict(ckpt['model'])
    y = model.analysis_transform(x)
    y_strings = model.entropy_bottleneck.compress(y)
    x_shape = x_cpu.shape[2:]
    y_shape = y.shape[2:] #y.shape: torch.Size([1, 32, 32, 32, 32])
    print("x_shape:",x_shape)
    print("y_shape:",y_shape)
    ret = {'string':y_strings, 'x_shape':x_shape, 'y_shape':y_shape}
    output = os.path.split(args.input_file)[-1][:-4]
    output_file = args.output_dir + output + '.bin'
    print(output_file)
    os.makedirs(args.output_dir, exist_ok=True)
    with gzip.open(output_file, "wb") as f:
        representation = compress(ret)
        f.write(representation)
import os
import numpy as np
import argparse
import torch
from compression_model import PCCModel
import pc_io #无框架
import multiprocessing
import gzip
from tqdm import tqdm
from glob import glob
import pc_error

np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TYPE = np.uint16
DTYPE = np.dtype(TYPE)
SHAPE_LEN = 3
def load_compressed_file(file):
    with gzip.open(file, "rb") as f:
        x_shape = np.frombuffer(f.read(DTYPE.itemsize * SHAPE_LEN), dtype=TYPE)
        y_shape = np.frombuffer(f.read(DTYPE.itemsize * SHAPE_LEN), dtype=TYPE)
        string = f.read()

        return x_shape, y_shape, string

def load_compressed_files(files, batch_size=32):
    files_len = len(files)

    with multiprocessing.Pool() as p:
        data = np.array(list(tqdm(p.imap(load_compressed_file, files, batch_size), total=files_len)))

    return data

def quantize_tensor(x):
    # x = tf.clip_by_value(x, 0, 1)
    x = torch.clamp(x, min=0, max=1)

    x = torch.round(x)
    # x = tf.cast(x, tf.uint8)
    x = x.int()
    return x.cpu().numpy()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='decompress.py',
        description='Decompress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--input_dir',type=str, default='/userhome/pcc_geo_cnn_v1/pcc_geo_cnn_master/pytorch/output/',
        help='Input directory.')
    parser.add_argument(
        '--output_dir',type=str, default='/userhome/pcc_geo_cnn_v1/pcc_geo_cnn_master/pytorch/output/',
        help='Output directory.')
    parser.add_argument(
        "--init_ckpt", type=str, default='/userhome/pcc_geo_cnn_v1/pcc_geo_cnn_master/pytorch/models/epoch_1_273.pth', dest="init_ckpt",
        help='initial checkpoint directory.')
    parser.add_argument(
        '--num_filters', type=int, default=32,
        help='Number of filters per layer.')

    args = parser.parse_args()
    input_glob = os.path.join(args.input_dir, "**/*.bin")
    files = np.array(glob(input_glob, recursive=True))
    for idx,input_file in enumerate(files):
        output = os.path.split(input_file)[-1][:-4]
        output_file = args.output_dir + output + '_rec.ply'
        print(output_file)
        compressed_data = load_compressed_files([input_file], 1)
        x_shape = compressed_data[0][0]
        y_shape = compressed_data[0][1]
        y_shape = [1,args.num_filters] + [int(s) for s in y_shape]

        compressed_strings = compressed_data[0][2]
        model = PCCModel(num_filters=args.num_filters)
        model = model.to(device)
        ckpt = torch.load(args.init_ckpt)
        model.load_state_dict(ckpt['model'])
        y_hat = model.entropy_bottleneck.decompress(compressed_strings, y_shape)
        x_hat = model.synthesis_transform(y_hat.to(device))
        # Crop away any extraneous padding on the bottom
        # or right boundaries.
        x_hat = x_hat[:, :, :x_shape[0], :x_shape[1], :x_shape[2]]
        x_hat_quant = quantize_tensor(x_hat)
        os.makedirs(args.output_dir, exist_ok=True)

        pa = np.argwhere(x_hat_quant[0][0]).astype('float32') #返回非0的数组元组的索引
        pc_io.write_df(output_file, pc_io.pa_to_df(pa))
        ori = '/userhome/pcc_geo_cnn_v1/pcc_geo_cnn_master/data/test_data/' + output + '.ply'
        print("res:",x_shape[0])
        pc_error.pc_error(ori, output_file, x_shape[0], normal=True, show=True)

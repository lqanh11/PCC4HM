import json
# import logging
from model_syntax import save_compressed_file #无框架
import os
import numpy as np
# import tensorflow.compat.v1 as tf
import torch
import argparse
import gzip
from tqdm import trange
# from model_configs import ModelConfigType
import pc_io #无框架
from octree_coding import partition_octree
from pc_metric import avail_opt_metrics, validate_opt_metrics #无框架
from pyntcloud import PyntCloud
from model_configs import C3PCompressionModelV2
from data_loader import points2voxels
from model_opt import compute_optimal_thresholds
from scipy.spatial import cKDTree
from octree_coding import departition_octree
from pc_metric import compute_metrics

np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def write_pcs(pcs, folder):
    os.makedirs(folder, exist_ok=True)
    for j, points in enumerate(pcs):
        pc_io.write_df(os.path.join(folder, f'{j}.ply'), pc_io.pa_to_df(points))

def get_normals_if(x, with_normals):
    return x[:, x.shape[1]-3:x.shape[1]] if with_normals else None

def select_best_per_opt_metric(binstr, x_hat_list, level, opt_metrics, points, resolution, with_normals,
                               opt_groups=('d1', 'd2')):
    """Selects best opt_metric for each opt_group

    :param binstr: octree binstr specification
    :param x_hat_list: a list of list of blocks for each opt_metric
    :param level: octree partitioning level
    :param opt_metrics: list of opt_metrics names
    :param points: original points for comparison
    :param resolution: original point cloud resolution
    :param with_normals: whether normals are available or not
    :param opt_groups: opt_metric prefixes used for grouping
    :return: metadata regarding best selections
    """
    assert len(opt_metrics) == len(x_hat_list), f'lengths of opt_metrics {len(opt_metrics)} and x_hat_list' +\
                                                f' {len(x_hat_list)} should be equal'
    # opt_metric groups
    om_groups = [[(x, y, i) for i, (x, y) in enumerate(zip(opt_metrics, x_hat_list))
                  if x.startswith(group)] for group in opt_groups] #len为2，x是指标，如d1_mse_inf，y是该指标下所有block的坐标集合，i是0,1
    bbox_min = [0, 0, 0]
    # Assume blocks of equal size
    bbox_max = [resolution] * 3 #resolution：1024 [1024, 1024, 1024]
    t1 = cKDTree(points[:, :3])
    metadata = []
    print(f'Processing metrics {opt_metrics} with groups {opt_groups}')
    for group, om_group in zip(opt_groups, om_groups):
        metric_key = f'{group}_psnr'
        if len(om_group) == 0:
            print(f'Group {group} : {metric_key} no data')
            continue
        om_names, cur_x_hat_list, indexes = zip(*om_group) #d1_mse_inf ，d1_mse_inf指标下所有block的坐标集合 ，0
        print("len(cur_x_hat_list):",len(cur_x_hat_list)) #1
        print("len(cur_x_hat_list[0]):",len(cur_x_hat_list[0])) #202
        cur_blocks_depart = [departition_octree(x, binstr, bbox_min, bbox_max, level) for x in cur_x_hat_list] #x相当于一次把所有blocks都传进去，这个没有仔细看
        print("len(cur_blocks_depart):",len(cur_blocks_depart)) #1
        print("len(cur_blocks_depart[0]):",len(cur_blocks_depart[0])) #202
        # print("cur_blocks_depart[0][:3]:",cur_blocks_depart[0][:3]) #[array([[295.,  17., 191.],...[319.,  63., 167.]]), array([[320.,  23., 191.],...[357.,  63., 173.]]), array([[300., 124., 189.],...[319., 127., 170.]])]
        cur_blocks_full = [np.vstack(x) for x in cur_blocks_depart] #所有block堆叠成一个
        print("len(cur_blocks_full):",len(cur_blocks_full)) #1
        print("len(cur_blocks_full[0]):",len(cur_blocks_full[0])) #771187
        # print("cur_blocks_full[0][:20]:",cur_blocks_full[0][:20]) #[[295.  17. 191.] ... [295.  33. 191.]]
        cur_metrics_full = [compute_metrics(points[:, :3], x, resolution - 1, p1_n=get_normals_if(points, with_normals),
                                            t1=t1) for x in cur_blocks_full]
        print("cur_metrics_full:",cur_metrics_full) #[{'d1_sum_AB': 104918.0, 'd1_sum_BA': 114964.0, 'd1_sum_max': 114964.0, 'd1_sum_mean': 109941.0, 'd1_mse_AB': 0.13847069583774915, 'd1_mse_BA': 0.14907408968252836, 'd1_mse': 0.14907408968252836, 'd1_psnr_AB': 73.55514647534262, 'd1_psnr_BA': 73.23470356127291, 'd1_psnr': 73.23470356127291, 'd2_sum_AB': 38122.8996478324, 'd2_sum_BA': 55431.859244623345, 'd2_sum_max': 55431.859244623345, 'd2_sum_mean': 46777.379446227875, 'd2_mse_AB': 0.05031457368219023, 'd2_mse_BA': 0.07187862249314803, 'd2_mse': 0.07187862249314803, 'd2_psnr_AB': 77.95178724900639, 'd2_psnr_BA': 76.40272776597378, 'd2_psnr': 76.40272776597378}]
        cur_metrics = [x[metric_key] for x in cur_metrics_full]
        local_best_idx = np.argmax(cur_metrics) #0
        print("indexes:",indexes) #(0,) (1,)
        best_idx = indexes[local_best_idx]
        data = {'idx': best_idx,
                'metrics': cur_metrics_full[local_best_idx],
                'x_hat_list': cur_x_hat_list[local_best_idx],
                'blocks_depart': cur_blocks_depart[local_best_idx],
                'blocks_full': cur_blocks_full[local_best_idx]}
        results_dict = dict(zip(om_names, [f"{x:.2f}" for x in cur_metrics])) #{'d1_mse_inf': '73.23'}
        print(f'Group {group} : {metric_key} best idx {best_idx} {opt_metrics[best_idx]}\n')
        metadata.append(data)
    return metadata

def compress_blocks(model,x_shape, blocks, binstr, points, resolution, level, with_normals=False,
                    opt_metrics=('d1_mse',), max_deltas=(np.inf,), fixed_threshold=False, debug=False):
    """Uses the compression model to compress a point cloud"""
    strings_list = []
    threshold_list = []
    debug_t_list = []
    x_hat_list = []
    # no_op = tf.no_op()
    for j, block in enumerate(blocks): #blocks:[202,1845,6]
        print(f'Compress block {j}/{len(blocks)}: start')
        # block_uint32 = block.astype(np.uint32) #[1845,6]
        voxels = points2voxels([block],x_shape[-1]).astype('float32') #[1,64,64,64]
        x_np = np.expand_dims(voxels,0)  #[1,1,64,64,64]
        x_cpu = torch.from_numpy(x_np) #(1, 1, 64, 64, 64)
        x = x_cpu.to(device) #转到GPU
        # x_val = sparse_to_dense(block_uint32, self.x.shape, self.data_format) #到这里 #有坐标的为1，其余为0
        print('Compress block: run session')

        y = model.analysis_transform(x) #y.shape: (1, 64, 8, 8, 8)
        z = model.hyper_analysis_transform(y) #z.shape: (1, 64, 4, 4, 4)
        z_tilde, _, _ = model.entropy_bottleneck(z, quantize_mode="symbols")
        sigma_tilde = model.hyper_synthesis_transform(z_tilde) #sigma_tilde.shape: (1, 64, 8, 8, 8)
        y_tilde, _ = model.conditional_bottleneck(y, scale=sigma_tilde, quantize_mode="symbols")
        z_strings = model.entropy_bottleneck.compress(z)
        x_hat = model.synthesis_transform(y_tilde)
        sigma_tilde = torch.clamp(sigma_tilde, min=args.scale_lower_bound)
        y_strings, y_min_vs, y_max_vs = model.conditional_bottleneck.compress(y, sigma_tilde)

        # fetches = [self.strings, self.x_hat, self.debug_tensors if debug else no_op]
        # strings, x_hat, debug_tensors = sess.run(fetches, feed_dict={self.x: x_val}) #这一步挺快的
        print('Compress block: session done')
        strings = [y_strings, z_strings]
        # strings = [s[0] for s in strings] #y_string
        x_hat = x_hat[0, 0, :, :, :].detach().cpu().numpy() # if self.data_format == 'channels_first' else x_hat[0, :, :, :, 0]
        x_hat = np.clip(x_hat, 0.0, 1.0) #x_hat.shape: (64, 64, 64)
        print('Compress block: compute optimal thresholds')
        normals = get_normals_if(block, with_normals) #取法向量
        n_thresholds=2 ** 8
        thresholds = np.linspace(0, 1.0, n_thresholds)
        opt_metrics_ret, best_thresholds = compute_optimal_thresholds(block, x_hat, thresholds, resolution, #这一步慢，到这里
                                                                      normals=normals, opt_metrics=opt_metrics,
                                                                      max_deltas=max_deltas, fixed_threshold=fixed_threshold)
        print('Compress block: done')
        x_hat_list.append([np.argwhere(x_hat > thresholds[t]).astype(np.float32) for t in best_thresholds]) #np.argwhere返回非0的数组元组的索引
        strings_list.append(strings)
        threshold_list.append(best_thresholds)
        # debug_t_list.append(debug_tensors)
    # block -> opt metric to opt metric -> block
    threshold_list = list(zip(*threshold_list)) #len为2，threshold_list[0]为第一个指标下的所有block最匹配的thresholds的idx集合
    x_hat_list = list(zip(*x_hat_list)) #len为2，x_hat_list[0]长度为202，为第一个指标下的所有block的坐标的集合
    metadata = select_best_per_opt_metric(binstr, x_hat_list, level, opt_metrics_ret, points, resolution, with_normals) #binstr是八叉树结构
    data_list = [list(zip(strings_list, threshold_list[x['idx']])) for x in metadata]
    return data_list, metadata, debug_t_list

def compress():
    args_resolution = 1024
    args.octree_level = 4
    if 'vox10' in args.input_files[0]:
        args_resolution = 1024
        args.octree_level = 4
    elif 'vox9' in args.input_files[0]:
        args_resolution = 512
        args.octree_level = 3
    elif 'vox11' in args.input_files[0]:
        args_resolution = 2048
        args.octree_level = 5
    assert args_resolution > 0, 'resolution must be positive'
    # assert args.data_format in ['channels_first', 'channels_last']
    with_normals = args.input_normals is not None #true
    validate_opt_metrics(args.opt_metrics, with_normals=with_normals)

    files_mult = 1
    if len(args.opt_metrics) > 1:
        files_mult *= len(args.opt_metrics) #2
        assert files_mult * len(args.input_files) == len(args.output_files) #output_files编码输出文件
        assert files_mult * len(args.input_normals) == len(args.output_files)
    else:
        assert files_mult * len(args.input_files) == len(args.output_files)
    decode_files = args.dec_files is not None
    if decode_files:
        assert files_mult * len(args.input_files) == len(args.dec_files)

    # assert args.model_config in ModelConfigType.keys()

    p_min, p_max, dense_tensor_shape = pc_io.get_shape_data(args_resolution, 'channels_first')#channels_first dense_tensor_shape: [   1 1024 1024 1024]
    points = pc_io.load_points(args.input_files, batch_size=args.read_batch_size) #points.shape: (1, 757691, 3)只有xyz
    if with_normals:
        normals = [PyntCloud.from_file(x).points[['nx', 'ny', 'nz']].values for x in args.input_normals]
        points = [np.hstack((p, n)) for p, n in zip(points, normals)]
    print("len(points):",len(points)) #len(points): 1  points[0].shape: (757691, 6)  points[0][0]: [255. 39. 291. 0.45551965 0.5938891 0.66317236]
    print('Performing octree partitioning')
    # Hardcode bbox_min
    bbox_min = [0, 0, 0]
    bbox_max = dense_tensor_shape[1:].copy()
    dense_tensor_shape[1:] = dense_tensor_shape[1:] // (2 ** args.octree_level)
    blocks_list, binstr_list = zip(*[partition_octree(p, bbox_min, bbox_max, args.octree_level) for p in points])
    print("len(blocks_list):",len(blocks_list)) #1
    print("len(blocks_list[0]):",len(blocks_list[0])) #202
    print("blocks_list[0][0].shape:",blocks_list[0][0].shape) #(1845, 6)
    print("blocks_list[0][1].shape:",blocks_list[0][1].shape) #(3125, 6)
    # print("blocks_list[0][0]:",blocks_list[0][0]) #见调试记录.txt
    # print("blocks_list[0][1]:",blocks_list[0][1])
    # print("blocks_list[0][2]:",blocks_list[0][2])
    print("len(binstr_list):",len(binstr_list)) #1
    print("len(binstr_list[0]):",len(binstr_list[0])) #67
    # print("binstr_list[0]:",binstr_list[0]) #[13, 254, 80, 255, 255,..., 240, 5, 12, 1]
    blocks_list_flat = [y for x in blocks_list for y in x]
    print(f'Processing resolution {args_resolution} with octree level {args.octree_level} resulting in '
                + f'dense_tensor_shape {dense_tensor_shape} and {len(blocks_list_flat)} blocks') #dense_tensor_shape [ 1 64 64 64]
    print("blocks_list_flat[0].shape:",blocks_list_flat[0].shape) #(1845, 6)
    batch_size = 1
    x_shape = np.concatenate(((batch_size,), dense_tensor_shape)) #x_shape: [ 1  1 64 64 64]

    # model = ModelConfigType[args.model_config].build() #这里返回的是一个类的实例对象
    model = C3PCompressionModelV2(num_filters=args.num_filters,scale_lower_bound=args.scale_lower_bound)
    # print("model:",model) #<model_types.CompressionModelV2 object at 0x7fa21742aa58>
    model = model.to(device)
    ckpt = torch.load(args.checkpoint_dir)
    model.load_state_dict(ckpt['model'])
    # model.compress(x_shape)

    # Checkpoints
    # saver = tf.train.Saver(keep_checkpoint_every_n_hours=1)
    # init = tf.global_variables_initializer()
    # tf_config = tf.ConfigProto()
    # tf_config.gpu_options.allow_growth = True
    # with tf.Session(config=tf_config) as sess:
    #     print('Init session')
    #     sess.run(init)

        # checkpoint = tf.train.latest_checkpoint(args.checkpoint_dir)
        # assert checkpoint is not None, f'Checkpoint {args.checkpoint_dir} was not found'
        # saver.restore(sess, checkpoint)

    for i in trange(len(args.input_files)): #trange进度条
        ori_file, cur_points, blocks, binstr = [x[i] for x in (args.input_files, points, blocks_list, binstr_list)] #blocks_list：[1,202,1845,6],这不是数组，1845不是固定的
        n_blocks = len(blocks) #202

        cur_output_files = [args.output_files[i*files_mult+j] for j in range(files_mult)] #2个
        if decode_files:
            cur_dec_files = [args.dec_files[i*files_mult+j] for j in range(files_mult)] #2个
        assert len(set(cur_output_files)) == len(cur_output_files), f'{cur_output_files} should have no duplicates'
        print(f'Starting {ori_file} to {", ".join(cur_output_files)} with {n_blocks} blocks')


        data_list, data, _ = compress_blocks(model,x_shape, blocks, binstr, cur_points, args_resolution,
                                                                args.octree_level, with_normals=with_normals,
                                                                opt_metrics=args.opt_metrics, max_deltas=args.max_deltas,
                                                                fixed_threshold=args.fixed_threshold, debug=False)
        

        assert len(data_list) == files_mult

        for j in range(len(cur_output_files)): #2个
            of, cur_data_list, cur_data = [x[j] for x in (cur_output_files, data_list, data)]
            # os.makedirs(os.path.split(of)[0], exist_ok=True)
            with gzip.open(of, "wb") as f:
                ret = save_compressed_file(binstr, cur_data_list, args_resolution, args.octree_level)
                f.write(ret)
            if decode_files:
                pc_io.write_df(cur_dec_files[j], pc_io.pa_to_df(cur_data['blocks_full']))
            with open(of + '.enc.metric.json', 'w') as f:
                json.dump(cur_data['metrics'], f, sort_keys=True, indent=4)
            if False:
                pc_io.write_df(of + '.enc.ply', pc_io.pa_to_df(cur_data['blocks_full']))

                write_pcs(blocks, of + '.ori.blocks')
                write_pcs(cur_data['x_hat_list'], of + '.enc.blocks')
                write_pcs(cur_data['blocks_depart'], of + '.enc.blocks.depart')
                # np.savez_compressed(of + '.enc.data.npz', data=cur_data_list, debug_t_list=debug_t_list)

        print(f'Finished {ori_file} to {", ".join(cur_output_files)} with {n_blocks} blocks')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='compress_octree.py',
        description='Compress a file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--input_files', default='/media/avitech/CODE/quocanhle/Point_Cloud_Compression/dataset/testdata/MVUB/sarah_vox9_frame0023.ply',
        help='Input files.')
    parser.add_argument(
        '--output_files', default='Sarah_vox9_0023_d1.ply.bin', #,'Sarah_vox9_0023_d2.ply.bin'
        help='Output files. If input normals are provided, specify two output files per input file.')
    parser.add_argument(
        '--input_normals', default='/media/avitech/CODE/quocanhle/Point_Cloud_Compression/dataset/testdata/MVUB/sarah_vox9_frame0023.ply',
        help='Input normals. If provided, two output paths are needed for each input file for D1 and D2 optimization.')
    parser.add_argument(
        '--dec_files', default='Sarah_vox9_0023_d1.ply.bin.ply',#,'Sarah_vox9_0023_d2.ply.bin.ply'
        help='Decoded files. Allows compression/decompression in a single execution. If input normals are provided, '
             + 'specify two decoded files per input file.') #这个有
    parser.add_argument(
        '--checkpoint_dir',default='/media/avitech/CODE/quocanhle/Point_Cloud_Compression/compression_frameworks/GEO_CNN/pcc_geo_cnn_v2_yehua/pytorch/models/3e-04/epoch_57_4009.pth',
        help='Directory where to save/load model checkpoints.')
    # parser.add_argument(
    #     '--model_config',
    #     help=f'Model used: {ModelConfigType.keys()}.', required=True) #c3p
    parser.add_argument(
        '--opt_metrics', nargs='+', default=['d1_mse'],#,'d2_mse'
        help=f'Optimization metrics used. Available: {avail_opt_metrics}')
    parser.add_argument(
        '--max_deltas', nargs='+', default=[np.inf], type=float,
        help=f'Max deltas tested during optimization.')
    parser.add_argument(
        '--fixed_threshold', default=False, action='store_true',
        help='Enable fixed thresholding.')
    parser.add_argument(
        '--read_batch_size', type=int, default=1,
        help='Batch size for parallel reading.')
    # parser.add_argument(
    #     '--resolution',
    #     type=int, help='Dataset resolution.', default=64)
    parser.add_argument(
        '--octree_level',
        type=int, help='Octree level.', default=4)
    parser.add_argument(
        '--num_filters', type=int, default=64,
        help='Number of filters per layer.')
    # parser.add_argument(
    #     '--data_format', default='channels_first',
    #     help='Data format used: channels_first or channels_last')
    # parser.add_argument(
    #     '--debug', default=False, action='store_true',
    #     help='Output debug data for point cloud.')
    parser.add_argument(
        "--scale_lower_bound", type=float, default=1e-5, dest="scale_lower_bound",
        help="lower bound of scale. 1e-5 or 1e-9")

    args = parser.parse_args()
    args.input_files = [args.input_files]
    args.output_files = [args.output_files]
    args.input_normals = [args.input_normals]
    args.dec_files = [args.dec_files]
    compress()

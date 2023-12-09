import numpy as np
import logging
from scipy.spatial.ckdtree import cKDTree
from pc_metric import validate_opt_metrics, compute_metrics

logger = logging.getLogger(__name__)


def build_points_threshold(x_hat, thresholds, len_block, max_delta=np.inf):
    pa_list = []
    for i, t in enumerate(thresholds):
        pa = np.argwhere(x_hat > t).astype('float32')
        if len(pa) == 0:
            break
        len_ratio = len(pa) / len_block
        if (1 / max_delta) < len_ratio < max_delta: #0<len_ratio<inf
            pa_list.append((i, pa))
    return pa_list


def compute_optimal_thresholds(block, x_hat, thresholds, resolution, normals=None, opt_metrics=['d1_mse'],
                               max_deltas=[np.inf], fixed_threshold=False):
    validate_opt_metrics(opt_metrics, with_normals=normals is not None)
    assert len(max_deltas) > 0
    best_thresholds = []
    ret_opt_metrics = [f'{opt_metric}_{max_delta}' for max_delta in max_deltas for opt_metric in opt_metrics] #ret_opt_metrics: ['d1_mse_inf', 'd2_mse_inf']
    if fixed_threshold:
        half_thr = len(thresholds) // 2
        half_pa = np.argwhere(x_hat > thresholds[half_thr]).astype('float32')
        logger.info(f'Fixed threshold {half_thr}/{len(thresholds)} with {len(half_pa)}/{len(block)} points (ratio {len(half_pa)/len(block):.2f})')
        return ret_opt_metrics, [half_thr] * len(max_deltas) * len(opt_metrics)

    pa_list = build_points_threshold(x_hat, thresholds, len(block)) #x_hat相当于解码后的点 x_hat[0,0,0]: 0.004279822 这里是针对thresholds里每个值都得到一个解码的结果，搞成一个list
    max_threshold_idx = len(thresholds) - 1
    if len(pa_list) == 0:
        return ret_opt_metrics, [max_threshold_idx] * len(opt_metrics)

    t1 = cKDTree(block[:, :3], balanced_tree=False) #对原始点建立二叉树
    pa_metrics = [compute_metrics(block[:, :3], pa, resolution - 1, p1_n=normals, t1=t1) for _, pa in pa_list] #这里慢，3-4s，针对每个阈值的解码点来计算，pa的长度不固定 pa_metrics：与不同阈值的解码点计算metrics后的集合

    log_message = f'Processing max_deltas {max_deltas} on block with {len(block)} points'
    for max_delta in max_deltas:
        if max_delta is not None:
            cur_pa_list = build_points_threshold(x_hat, thresholds, len(block), max_delta) #前面pa_list是指定inf，这里的max_delta可能是其他值，如果不是inf，cur_pa_list会是pa_list的子集
            if len(cur_pa_list) > 0: #len(cur_pa_list): 246
                # print("len(cur_pa_list[0]):",len(cur_pa_list[0])) #2
                # print("cur_pa_list[1]:",cur_pa_list[1]) #(1, array([[ 0.,  0.,  0.],[ 0., 24., 62.],...,[63., 63., 49.]], dtype=float32))
                idx_mask = [x[0] for x in cur_pa_list] #len(idx_mask): 246    idx_mask[-1]: 245
                cur_pa_metrics = [pa_metrics[i] for i in idx_mask] #len(pa_metrics): 246 len(cur_pa_metrics): 246 max_delta=inf的情况下，这几步操作等于啥也没干
            else:
                cur_pa_list = pa_list
                cur_pa_metrics = pa_metrics
        else:
            cur_pa_list = pa_list
            cur_pa_metrics = pa_metrics
        log_message += f'\n{len(cur_pa_list)}/{len(thresholds)} thresholds eligible for max_delta {max_delta}'
        for opt_metric in opt_metrics:
            best_threshold_idx = np.argmin([x[opt_metric] for x in cur_pa_metrics]) #最小值的索引，是一个数
            cur_best_metric = cur_pa_metrics[best_threshold_idx][opt_metric] #最小的metrics

            # Check for failure scenarios
            mean_point_metric = compute_metrics(block[:, :3],
                                                np.round(np.mean(block[:, :3], axis=0))[np.newaxis, :],
                                                resolution - 1, p1_n=normals, t1=t1)[opt_metric]
            # In case a single point is better than the network output, this is a failure case
            # Do not output any points
            if cur_best_metric > mean_point_metric:
                best_threshold_idx = max_threshold_idx
                final_idx = best_threshold_idx
                log_message += f', {opt_metric} {final_idx} 0/{len(block)}, metric {cur_best_metric:.2e} > mean point metric {mean_point_metric:.2e}'
            else:
                final_idx = cur_pa_list[best_threshold_idx][0] #这个是thresholds的最佳index
                cur_n_points = len(cur_pa_list[best_threshold_idx][1]) #最佳适配的解码结果的点的个数
                log_message += f', {opt_metric} {final_idx} {cur_n_points}/{len(block)} points (ratio {cur_n_points/len(block):.2f}) {cur_best_metric :.2e} < mean point metric {mean_point_metric:.2e}'
            best_thresholds.append(final_idx)
    logger.info(log_message)
    assert len(ret_opt_metrics) == len(best_thresholds)

    return ret_opt_metrics, best_thresholds
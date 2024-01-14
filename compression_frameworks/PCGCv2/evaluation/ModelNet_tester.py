import os, sys, time, logging

sys.path.append('../')

from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME

from loss import get_bce, get_bits, get_metrics
from data_utils import array2vector, istopk, sort_spare_tensor, load_sparse_tensor, scale_sparse_tensor
from data_utils import write_ply_ascii_geo, read_ply_ascii_geo

from gpcc import gpcc_encode, gpcc_decode
from pc_error import pc_error

import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Tester():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        # self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        # self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {'bce':[], 'bces':[], 'bpp':[], 'bits': [], 'metrics':[]}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log_test.txt'))
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
        handler.setFormatter(formatter)
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self):
        """selectively load model
        """
        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))
            
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  

        return
    
    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        
        self.model.eval()

        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for idx, batch in enumerate(tqdm(dataloader)):
            # data
            pc_filedir = batch["dense_pc_file_dir"][0]
            basename = os.path.split(pc_filedir)[-1].split('.')[0]

            save_basename = os.path.join(self.config.outdir, basename)

            x = ME.SparseTensor(features=batch["dense_features"],
                                coordinates=batch["dense_coordinates"],
                                device=device)


            ## encoder_side
 
            start_time = time.time()

            y_list = self.model.encoder(x)
            y = sort_spare_tensor(y_list[0])
            num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]
            with open(save_basename+'_num_points.bin', 'wb') as f:
                f.write(np.array(num_points, dtype=np.int32).tobytes())

            ### coordinate coder
            y_C = (y.C//y.tensor_stride[0]).detach().cpu()[:,1:].numpy().astype('int')
            write_ply_ascii_geo(filedir=save_basename+'_latentspace_coods.ply', coords=y_C)
            gpcc_encode(save_basename+'_latentspace_coods.ply', save_basename+'_C.bin')
            os.system('rm '+save_basename+'_latentspace_coods.ply')

            ### features coder
            strings, min_v, max_v = self.model.entropy_bottleneck.compress(y.F.cpu())
            shape = y.F.shape
            
            with open(save_basename+'_F.bin', 'wb') as fout:
                fout.write(strings)
            with open(save_basename+'_H.bin', 'wb') as fout:
                fout.write(np.array(shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(min_v), dtype=np.int8).tobytes())
                fout.write(np.array(min_v, dtype=np.float32).tobytes())
                fout.write(np.array(max_v, dtype=np.float32).tobytes())

            time_enc = round(time.time() - start_time, 3)

            ## decoder part

            start_time = time.time()

            ### deocde coordinates
            gpcc_decode(save_basename+'_C.bin', save_basename+'_latentspace_coods_decoded.ply')
            y_C_hat = read_ply_ascii_geo(save_basename+'_latentspace_coods_decoded.ply')
            os.system('rm '+save_basename+'_latentspace_coods_decoded.ply')

            y_C_hat = torch.cat((torch.zeros((len(y_C_hat),1)).int(), torch.tensor(y_C_hat).int()), dim=-1)
            indices_sort = np.argsort(array2vector(y_C_hat, y_C_hat.max()+1))
            y_C_hat = y_C_hat[indices_sort]

            ### decode features
            with open(save_basename+'_F.bin', 'rb') as fin:
                strings_hat = fin.read()
            with open(save_basename+'_H.bin', 'rb') as fin:
                shape_hat = np.frombuffer(fin.read(4*2), dtype=np.int32)
                len_min_v_hat = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                min_v_hat = np.frombuffer(fin.read(4*len_min_v_hat), dtype=np.float32)[0]
                max_v_hat = np.frombuffer(fin.read(4*len_min_v_hat), dtype=np.float32)[0]
                
            y_F_hat = self.model.entropy_bottleneck.decompress(strings_hat, min_v_hat, max_v_hat, shape_hat, channels=shape_hat[-1])

            y_hat = ME.SparseTensor(features=y_F_hat, coordinates=y_C_hat*8,
                            tensor_stride=8, device=device)
            # decode label
            with open(save_basename+'_num_points.bin', 'rb') as fin:
                num_points_hat = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
                num_points_hat[-1] = int(num_points_hat[-1])# update
                num_points_hat = [[num] for num in num_points_hat]

            out_cls_list, x_dec = self.model.decoder(y_hat, nums_list=num_points_hat, ground_truth_list=[None]*3, training=False)

            time_dec = round(time.time() - start_time, 3)

            write_ply_ascii_geo(save_basename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])


            # Evaluate    
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip([out_cls_list[2]], [x]):
                curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce 
                bce_list.append(curr_bce.item())

            bits = np.array([os.path.getsize(save_basename + postfix)*8 \
                                for postfix in ['_C.bin', '_F.bin', '_H.bin', '_num_points.bin']])

            bpp = bits / float(x.__len__())
            bits_avg = bits / len(num_points_hat[2])

            metrics = []
            for out_cls, ground_truth in zip([out_cls_list[2]], [x]):
                metrics.append(get_metrics(out_cls, ground_truth))

            # record
            self.record_set['bce'].append(bce.item())
            self.record_set['bces'].append(bce_list)
            self.record_set['bpp'].append(sum(bpp).item())
            self.record_set['bits'].append(sum(bits_avg).item())
            self.record_set['metrics'].append(metrics)

            results = {}
            results["num_points(input)"] = len(x)
            results["num_points(output)"] = len(x_dec)
            results["resolution"] = self.config.resolution
            results["bits"] = sum(bits).round(3)
            results["bpp"] = sum(bpp).round(3)
            results["bpp(coords)"] = bpp[0]
            results["bpp(feats)"] = bpp[1]
            results["time(enc)"] = time_enc
            results["time(dec)"] = time_dec

            df_results = pd.DataFrame([results])

            csv_name = os.path.join(save_basename +'_general.csv')
            df_results.to_csv(csv_name, index=False)
            print('Wrile general results to: \t', csv_name)

            # distortion
            pc_error_metrics = pc_error(pc_filedir, save_basename+'_dec.ply', 
                                    res=self.config.resolution, normal=True, show=False)
            results = pc_error_metrics
            csv_name = os.path.join(save_basename +'_distortion.csv')
            results.to_csv(csv_name, index=False)
            print('Wrile distortion results to: \t', csv_name)

            torch.cuda.empty_cache()# empty cache.

        self.record(main_tag=main_tag, global_step=self.epoch)

        return

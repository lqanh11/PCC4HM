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
        self.epoch = 0
        self.record_set = {
                        'bits': [],
                           }
        self.accuracy_set = {'number_correct': [], 'total_number':[]}
        

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

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))
            
        accuracy_cls = sum(self.accuracy_set['number_correct']) / sum(self.accuracy_set['total_number'])
        self.logger.info('ACC_CLS'+': '+str(np.round(accuracy_cls, 4).tolist()))

        avg_bits = self.record_set['bits'] 
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  
        self.accuracy_set = {'number_correct': [], 'total_number':[]}

        return avg_bits, accuracy_cls
    
    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        
        self.model.eval()
        labels_list, preds_list = [], []

        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for idx, batch in enumerate(tqdm(dataloader)):
            # data
            pc_filedir = batch["dense_pc_file_dir"][0]
            basename = os.path.split(pc_filedir)[-1].split('.')[0]

            save_basename = os.path.join(self.config.outdir, basename)

            x = ME.SparseTensor(features=batch["dense_features"],
                                coordinates=torch.floor(batch["dense_coordinates"]).int(),
                                device=device)
            labels = batch["labels"]

            ## encoder_side
 
            start_time = time.time()

            y_list = self.model.encoder(x)
            y = sort_spare_tensor(y_list[0])
            num_points = [len(ground_truth) for ground_truth in y_list[1:] + [x]]

            y_q, _ = self.model.get_likelihood(y, quantize_mode="symbols")
            # classification
            out_cls_backbone = self.model.classifier_backbone(y_q)

            ### features coder
            base_strings, base_min_v, base_max_v = self.model.entropy_bottleneck_b.compress(out_cls_backbone[:,:,0])
            base_shape = out_cls_backbone[:,:,0].shape
            
            with open(save_basename+'_F.bin', 'wb') as fout:
                fout.write(base_strings)
            with open(save_basename+'_H.bin', 'wb') as fout:
                fout.write(np.array(base_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(base_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(base_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(base_max_v, dtype=np.float32).tobytes())

            time_enc = round(time.time() - start_time, 3)

            ## decoder part
            start_time = time.time()
            ### decode features
            with open(save_basename+'_F.bin', 'rb') as fin:
                strings_hat = fin.read()
            with open(save_basename+'_H.bin', 'rb') as fin:
                shape_hat = np.frombuffer(fin.read(4*2), dtype=np.int32)
                len_min_v_hat = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                min_v_hat = np.frombuffer(fin.read(4*len_min_v_hat), dtype=np.float32)[0]
                max_v_hat = np.frombuffer(fin.read(4*len_min_v_hat), dtype=np.float32)[0]
                
            out_cls_backbone_q = self.model.entropy_bottleneck_b.decompress(strings_hat, min_v_hat, max_v_hat, shape_hat, channels=shape_hat[-1])

            logits = self.model.classifier_mlp(torch.unsqueeze(out_cls_backbone_q.to(device),-1))

            time_dec = round(time.time() - start_time, 3)

            # Evaluate    
            bits = np.array([os.path.getsize(save_basename + postfix)*8 \
                                for postfix in ['_F.bin', '_H.bin']])

            bpp = bits / float(x.__len__())

            self.record_set['bits'].append(sum(bits))

            # statistic
            logits = logits
            preds_class = torch.argmax(logits, 1)
            labels_class = labels.to(device)
            
            labels_list.append(labels.cpu().numpy())
            preds_list.append(preds_class.cpu().numpy())

            running_corrects_class = torch.sum(preds_class == labels_class)

            self.accuracy_set['number_correct'].append(running_corrects_class.item())
            self.accuracy_set['total_number'].append(labels.size(0))

            # record
            results = {}
            results["num_points(input)"] = len(x)
            results["resolution"] = self.config.resolution
            results["bits"] = sum(bits).round(3)
            results["bpp"] = sum(bpp).round(3)
            results["bpp(coords)"] = 0
            results["bpp(feats)"] = sum(bpp)
            results["time(enc)"] = time_enc
            results["time(dec)"] = time_dec

            df_results = pd.DataFrame([results])

            csv_name = os.path.join(save_basename +'_general.csv')
            df_results.to_csv(csv_name, index=False)
            # print('Wrile general results to: \t', csv_name)

            torch.cuda.empty_cache()# empty cache.

        avg_bits, accuracy_cls = self.record(main_tag=main_tag, global_step=self.epoch)

        return avg_bits, accuracy_cls

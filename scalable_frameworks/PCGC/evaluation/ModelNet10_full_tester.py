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
        # self.load_state_dict()
        self.epoch = 0
        self.record_set = {'bce':[], 'bces':[], 'bpp':[], 'bits': [], 'bits_b': [], 'bits_e': [], 'metrics':[]}
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
            with open(save_basename+'_num_points.bin', 'wb') as f:
                f.write(np.array(num_points, dtype=np.int32).tobytes())

            y_q, _ = self.model.get_likelihood(y, quantize_mode="symbols")

            ### coordinate coder
            
            y_C = (torch.div(y.C, y.tensor_stride[0], rounding_mode='trunc')).detach().cpu()[:,1:].numpy().astype('int')
            write_ply_ascii_geo(filedir=save_basename+'_latentspace_coods.ply', coords=y_C)
            gpcc_encode(save_basename+'_latentspace_coods.ply', save_basename+'_C.bin')
            os.system('rm '+save_basename+'_latentspace_coods.ply')

            ### base features coder
            out_cls_backbone = self.model.classifier_backbone(y_q)
            base_strings, base_min_v, base_max_v = self.model.entropy_bottleneck_b.compress(out_cls_backbone[:,:,0])
            base_shape = out_cls_backbone[:,:,0].shape
            
            with open(save_basename+'_base_F.bin', 'wb') as fout:
                fout.write(base_strings)
            with open(save_basename+'_base_H.bin', 'wb') as fout:
                fout.write(np.array(base_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(base_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(base_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(base_max_v, dtype=np.float32).tobytes())
            ### enhanment features coder
            
            #### base side
            out_cls_backbone_q, _ = self.model.get_likelihood_b(out_cls_backbone[:,:,0], quantize_mode="symbols")
            features_for_reconstruction = self.model.reconstruction_gain(torch.unsqueeze(out_cls_backbone_q,-1))[:,:,0]
            
            nums_latentspace = [[len(C) for C in latentspace.decomposed_coordinates] for latentspace in [y]]
            with open(save_basename+'_num_latentspace.bin', 'wb') as f:
                f.write(np.array(nums_latentspace, dtype=np.int32).tobytes())

            for index in range(len(nums_latentspace[0])):
                if index == 0:
                    value_a_channel = features_for_reconstruction[index,:]
                    tensor_2D = value_a_channel.repeat(nums_latentspace[0][index], 1)
                    tensor_all = tensor_2D
                else:
                    value_a_channel = features_for_reconstruction[index,:]
                    tensor_2D = value_a_channel.repeat(nums_latentspace[0][index], 1)
                    tensor_all = torch.cat((tensor_all, tensor_2D), 0)
            
            sparse_tensor_for_reconstruction = ME.SparseTensor(
                features=tensor_all, 
                coordinate_map_key=y_q.coordinate_map_key, 
                coordinate_manager=y_q.coordinate_manager, 
                device=y_q.device)
            y_b = self.model.reconstruction_backbone(sparse_tensor_for_reconstruction)

            #### calculate residual
            y_r_features =  y_q.F - y_b.F
            y_r = ME.SparseTensor(
                features=y_r_features, 
                coordinate_map_key=y_b.coordinate_map_key, 
                coordinate_manager=y_b.coordinate_manager, 
                device=y_b.device)
            z_r = self.model.analysis_residual(y_r)
            enhanment_strings, enhanment_min_v, enhanment_max_v = self.model.entropy_bottleneck_e.compress(z_r.F.cpu())
            enhanment_shape = z_r.F.shape
            
            with open(save_basename+'_enhanment_F.bin', 'wb') as fout:
                fout.write(enhanment_strings)
            with open(save_basename+'_enhanment_H.bin', 'wb') as fout:
                fout.write(np.array(enhanment_shape, dtype=np.int32).tobytes())
                fout.write(np.array(len(enhanment_min_v), dtype=np.int8).tobytes())
                fout.write(np.array(enhanment_min_v, dtype=np.float32).tobytes())
                fout.write(np.array(enhanment_max_v, dtype=np.float32).tobytes())


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
            ## base brach
            with open(save_basename+'_base_F.bin', 'rb') as fin:
                base_strings_hat = fin.read()
            with open(save_basename+'_base_H.bin', 'rb') as fin:
                base_shape_hat = np.frombuffer(fin.read(4*2), dtype=np.int32)
                base_len_min_v_hat = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                base_min_v_hat = np.frombuffer(fin.read(4*base_len_min_v_hat), dtype=np.float32)[0]
                base_max_v_hat = np.frombuffer(fin.read(4*base_len_min_v_hat), dtype=np.float32)[0]
                
            out_cls_backbone_q_hat = self.model.entropy_bottleneck_b.decompress(base_strings_hat, base_min_v_hat, base_max_v_hat, base_shape_hat, channels=base_shape_hat[-1])
            
            logits = self.model.classifier_mlp(torch.unsqueeze(out_cls_backbone_q_hat.to(device),-1))
            
            features_for_reconstruction_hat = self.model.reconstruction_gain(torch.unsqueeze(out_cls_backbone_q_hat.to(device),-1))[:,:,0]

            with open(save_basename+'_num_latentspace.bin', 'rb') as fin:
                nums_latentspace_hat = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
            nums_latentspace_hat = [nums_latentspace_hat]

            for index in range(len(nums_latentspace_hat[0])):
                if index == 0:
                    value_a_channel = features_for_reconstruction_hat[index,:]
                    tensor_2D = value_a_channel.repeat(nums_latentspace_hat[0][index], 1)
                    tensor_all_hat = tensor_2D
                else:
                    value_a_channel = features_for_reconstruction_hat[index,:]
                    tensor_2D = value_a_channel.repeat(nums_latentspace_hat[0][index], 1)
                    tensor_all_hat = torch.cat((tensor_all_hat, tensor_2D), 0)
            
            sparse_tensor_for_reconstruction_hat = ME.SparseTensor(
                features=tensor_all_hat, 
                coordinates=y_C_hat*8,
                tensor_stride=8, device=device)
            y_b_hat = self.model.reconstruction_backbone(sparse_tensor_for_reconstruction_hat)

            ### residual branch
            with open(save_basename+'_enhanment_F.bin', 'rb') as fin:
                enhanment_strings_hat = fin.read()
            with open(save_basename+'_enhanment_H.bin', 'rb') as fin:
                enhanment_shape_hat = np.frombuffer(fin.read(4*2), dtype=np.int32)
                enhanment_len_min_v_hat = np.frombuffer(fin.read(1), dtype=np.int8)[0]
                enhanment_min_v_hat = np.frombuffer(fin.read(4*enhanment_len_min_v_hat), dtype=np.float32)[0]
                enhanment_max_v_hat = np.frombuffer(fin.read(4*enhanment_len_min_v_hat), dtype=np.float32)[0]

            z_r_q_hat_F = self.model.entropy_bottleneck_e.decompress(enhanment_strings_hat, enhanment_min_v_hat, enhanment_max_v_hat, enhanment_shape_hat, channels=enhanment_shape_hat[-1])
            z_r_q_hat = ME.SparseTensor(
                features=z_r_q_hat_F, 
                coordinates=y_C_hat*8,
                tensor_stride=8, device=device)

            y_r_hat = self.model.systhesis_residual(z_r_q_hat)
            y_scalable_features = y_r_hat.F + y_b_hat.F
            y_scalable = ME.SparseTensor(
                features=y_scalable_features,
                coordinate_map_key=y_r_hat.coordinate_map_key, 
                coordinate_manager=y_r_hat.coordinate_manager, 
                device=y_r_hat.device)

            # decode label
            with open(save_basename+'_num_points.bin', 'rb') as fin:
                num_points_hat = np.frombuffer(fin.read(4*3), dtype=np.int32).tolist()
                num_points_hat[-1] = int(num_points_hat[-1])# update
                num_points_hat = [[num] for num in num_points_hat]

            out_cls_list, x_dec = self.model.decoder(y_scalable, nums_list=num_points_hat, ground_truth_list=[None]*3, training=False)

            time_dec = round(time.time() - start_time, 3)

            write_ply_ascii_geo(save_basename+'_dec.ply', x_dec.C.detach().cpu().numpy()[:,1:])

            # statistic
            logits = logits
            preds_class = torch.argmax(logits, 1)
            labels_class = labels.to(device)
            
            labels_list.append(labels.cpu().numpy())
            preds_list.append(preds_class.cpu().numpy())

            running_corrects_class = torch.sum(preds_class == labels_class)

            self.accuracy_set['number_correct'].append(running_corrects_class.item())
            self.accuracy_set['total_number'].append(labels.size(0))
            # Evaluate    
            bce, bce_list = 0, []
            for out_cls, ground_truth in zip([out_cls_list[2]], [x]):
                curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                bce += curr_bce 
                bce_list.append(curr_bce.item())

            bits_b = np.array([os.path.getsize(save_basename + postfix)*8 \
                                for postfix in ['_base_F.bin', '_base_H.bin']])
            bits_e = np.array([os.path.getsize(save_basename + postfix)*8 \
                                for postfix in ['_C.bin', '_num_latentspace.bin', '_enhanment_F.bin', '_enhanment_H.bin', '_num_points.bin']])
            bits = sum(bits_b) + sum(bits_e)

            bpp = bits / float(x.__len__())
           
            metrics = []
            for out_cls, ground_truth in zip([out_cls_list[2]], [x]):
                metrics.append(get_metrics(out_cls, ground_truth))

            # record
            self.record_set['bce'].append(bce.item())
            self.record_set['bces'].append(bce_list)
            self.record_set['bpp'].append(bpp.item())
            self.record_set['bits'].append(bits.item())
            self.record_set['bits_b'].append(bits_b)
            self.record_set['bits_e'].append(bits_e)
            self.record_set['metrics'].append(metrics)

            results = {}
            results["num_points(input)"] = len(x)
            results["num_points(output)"] = len(x_dec)
            results["resolution"] = self.config.resolution
            results["bits"] = bits.round(3)
            results["bpp"] = bpp.round(3)
            results["bits(base)"] = sum(bits_b).round(3)
            results["bits(enhanment)"] = sum(bits_e).round(3)
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

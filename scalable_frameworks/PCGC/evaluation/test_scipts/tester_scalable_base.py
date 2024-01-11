import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
import MinkowskiEngine as ME
import torch.nn.functional as F
import sklearn.metrics as metrics

from loss import get_bce, get_CE_loss, get_mse, get_bits, get_metrics
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from tensorboardX import SummaryWriter


class Test_Load_All():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        # self.logger.info(model)
        # self.load_state_dict()
        self.epoch = 0
        self.record_set = {
                        #    'mse':[],
                        #    'mses':[],
                           'ce_cls':[], 
                        #    'bpp_e':[], 
                           'bits_b': [],
                           'bits_b_all': [],
                        #    'bits_e': [],
                          }
        self.accuracy_set = {'number_correct': [], 'total_number':[]}

    def getlogger(self, logdir):
        logger = logging.getLogger(__name__)
        logger.setLevel(level = logging.INFO)
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
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

    def save_model(self):
        torch.save({'model': self.model.state_dict()}, 
            os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth'))
        return os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth')

    def set_optimizer(self):
        params_lr_list = []
        for module_name in self.model._modules.keys():
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step):
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
        for k, v in self.record_set.items(): 
            self.record_set[k]=np.mean(np.array(v), axis=0)
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))
            
            ## TensorbroadX visulization
            if k != 'bits_b_all':
                if k == 'metrics':
                    self.writer.add_scalar(f'{main_tag}/PRECISION', np.round(v[2][0], 4), self.epoch)
                    self.writer.add_scalar(f'{main_tag}/RECALL', np.round(v[2][1], 4), self.epoch)
                    self.writer.add_scalar(f'{main_tag}/IoU', np.round(v[2][2], 4), self.epoch)
                else:
                    self.writer.add_scalar(f'{main_tag}/{k}', np.round(v, 4), self.epoch)
            
        accuracy_cls = sum(self.accuracy_set['number_correct']) / sum(self.accuracy_set['total_number'])
        self.logger.info('ACC_CLS'+': '+str(np.round(accuracy_cls, 4).tolist()))
        self.writer.add_scalar(f'{main_tag}/acc_cls', np.round(accuracy_cls, 4), self.epoch)
            
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  
        self.accuracy_set = {'number_correct': [], 'total_number':[]}

        return 
    
    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        with torch.no_grad():
            self.logger.info('Testing Files length:' + str(len(dataloader)))

            labels_list, preds_list = [], []
            self.model.eval()
            for idx, batch in enumerate(tqdm(dataloader)):
                
                # data 
                y_q = ME.SparseTensor(features=batch["sparse_features"], 
                                      coordinates=batch["sparse_coordinates"]*8, 
                                      tensor_stride=8,
                                      device=device)
                labels = batch["labels"]
                num_points = np.transpose(np.array(batch["num_points"]))

                # # Forward.
                ## encoder part
                z = self.model.adapter(y_q)

                base_strings, base_min_v, base_max_v = self.model.entropy_bottleneck_b.compress(z.F)
                base_shape = z.F.shape
                with open('base_F.bin', 'wb') as fout:
                    fout.write(base_strings)
                with open('base_H.bin', 'wb') as fout:
                    fout.write(np.array(base_shape, dtype=np.int32).tobytes())
                    fout.write(np.array(len(base_min_v), dtype=np.int8).tobytes())
                    fout.write(np.array(base_min_v, dtype=np.float32).tobytes())
                    fout.write(np.array(base_max_v, dtype=np.float32).tobytes())

                z_q, _ = self.model.get_likelihood_b(z, quantize_mode="symbols")

                y_q_hat = self.model.latentspacetransform(z_q)
                
                # y_b = self.model.transpose_adapter(z_q)

                # y_r_features =  y_q.F - y_b.F
                # y_r = ME.SparseTensor(
                #     features=y_r_features, 
                #     coordinate_map_key=y_b.coordinate_map_key, 
                #     coordinate_manager=y_b.coordinate_manager, 
                #     device=y_b.device)
                # z_r = self.model.analysis_residual(y_r)
                # z_r_q, likelihood_e = self.model.get_likelihood_e(z_r, quantize_mode="symbols")
                
                ## decoder part
                logits = self.model.classifier(y_q_hat)
                # y_r_hat = self.model.systhesis_residual(z_r_q)
                # y_scalable_features = y_r_hat.F + y_b.F
                # y_scalable = ME.SparseTensor(
                #     features=y_scalable_features,
                #     coordinate_map_key=y_r_hat.coordinate_map_key, 
                #     coordinate_manager=y_r_hat.coordinate_manager, 
                #     device=y_r_hat.device)
                
                # # Decoder
                # out_cls_list, out = self.model.decoder(y_scalable, num_points, [None]*3, False)

                # mse, mse_list = 0, []
                # for out_cls, ground_truth in zip([y_scalable], [y_q]):
                #     curr_mse = get_mse(out_cls, ground_truth)
                #     mse += curr_mse
                #     mse_list.append(curr_mse.item())

                bits_b = np.array([
                                    os.path.getsize(batch["compression_files_dir"][0][0])*8,
                                    os.path.getsize('base_F.bin')*8,
                                    os.path.getsize('base_H.bin')*8,
                                   ])
                bits_b_avg = bits_b / len(num_points[2])
                # bits_e = get_bits(likelihood_e)
                # bits_e_avg = bits_e / len(num_points[2])

                # bpp_e = bits_e / float(out.__len__())

                ## CE classification
                ce_classification = get_CE_loss(logits, labels.to(device)) 
                # statistics
                logits = logits
                preds_class = torch.argmax(logits, 1)
                labels_class = labels.to(device)

                labels_list.append(labels.cpu().numpy())
                preds_list.append(preds_class.cpu().numpy())

                running_corrects_class = torch.sum(preds_class == labels_class)

                # record

                # self.record_set['mse'].append(mse.item())
                # self.record_set['mses'].append(mse_list)
                self.record_set['ce_cls'].append(ce_classification.item())
                # self.record_set['bpp_e'].append(bpp_e.item())
                self.record_set['bits_b'].append(sum(bits_b_avg))
                self.record_set['bits_b_all'].append(bits_b_avg)
                # self.record_set['bits_e'].append(bits_e_avg.item())
                self.accuracy_set['number_correct'].append(running_corrects_class.item())
                self.accuracy_set['total_number'].append(labels.size(0))

                torch.cuda.empty_cache()# empty cache.

            self.record(main_tag=main_tag, global_step=self.epoch)
            accuracy = metrics.accuracy_score(np.concatenate(labels_list), np.concatenate(preds_list))
            self.logger.info(f"Test accuracy: {accuracy}")

        return 
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


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.writer = SummaryWriter(log_dir=config.logdir)

        self.model = model.to(device)
        # self.logger.info(model)
        self.load_state_dict()
        self.epoch = 0
        self.record_set = {
            # 'bce':[],
            # 'mse':[],
            # 'bces':[],
            # 'mses':[],
            'ce_cls':[], 
            # 'bpp':[], 
            # 'bpp_e':[], 
            # 'bits': [],
            'bits_b': [],
            # 'bits_e': [],
            'sum_loss':[], 
            # 'metrics':[]
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
            if k != 'bces':
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
            for idx, (coords, coords_fix_pts, feats, feats_fix_pts, labels) in enumerate(tqdm(dataloader)):
                # data
                x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)

                # # Forward.
                out_set = self.model(x, training=False)
                # loss    
                # bce, bce_list = 0, []
                # for out_cls, ground_truth in zip(out_set['out_cls_list'], out_set['ground_truth_list']):
                #     curr_bce = get_bce(out_cls, ground_truth)/float(x.__len__())
                #     bce += curr_bce 
                #     bce_list.append(curr_bce.item())
                
                # mse, mse_list = 0, []
                # for out_cls, ground_truth in zip(out_set['y_b'], out_set['y']):
                #     curr_mse = get_mse(out_cls, ground_truth)
                #     mse += curr_mse
                #     mse_list.append(curr_mse.item())

                bits_b = get_bits(out_set['likelihood_b'])
                bits_b_avg = bits_b / len(out_set['nums_list'][2])

                ## CE classification
                ce_classification = get_CE_loss(out_set['logits'], labels.to(device)) 
                # sum_loss = self.config.alpha * ce_classification + self.config.gamma * mse + self.config.beta * (bits_b_avg)
                sum_loss = self.config.alpha * ce_classification + self.config.beta * (bits_b_avg)
                
                ## loss for base branch
                # sum_loss = self.config.alpha * ce_classification + self.config.beta * bits_b_avg
                # sum_loss = self.config.alpha * mse + self.config.beta * bpp_e
                # sum_loss = ce_classification

                # statistics
                logits = out_set['logits']
                preds_class = torch.argmax(logits, 1)
                labels_class = labels.to(device)

                labels_list.append(labels.cpu().numpy())
                preds_list.append(preds_class.cpu().numpy())

                running_corrects_class = torch.sum(preds_class == labels_class)

                # record
                # self.record_set['mse'].append(mse.item())
                # self.record_set['mses'].append(mse_list)
                self.record_set['ce_cls'].append(ce_classification.item())
                self.record_set['bits_b'].append(bits_b_avg.item())
                self.record_set['sum_loss'].append(sum_loss.item())

                self.accuracy_set['number_correct'].append(running_corrects_class.item())
                self.accuracy_set['total_number'].append(labels.size(0))

                torch.cuda.empty_cache()# empty cache.

            self.record(main_tag=main_tag, global_step=self.epoch)
            accuracy = metrics.accuracy_score(np.concatenate(labels_list), np.concatenate(preds_list))
            self.logger.info(f"Test accuracy: {accuracy}")

        return sum_loss.item()

    def train(self, dataloader, params_to_train):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))

        for name, param in self.model.named_parameters():
            # Set True only for params in the list 'params_to_train'
            decomposed_name = name.split(".")
            if(decomposed_name[0] in params_to_train):
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for batch_step, (coords, coords_fix_pts, feats, feats_fix_pts, labels) in enumerate(tqdm(dataloader)):
            self.optimizer.zero_grad()
            # data
            x = ME.SparseTensor(features=feats.float(), coordinates=coords, device=device)
            
            # labels = torch.from_numpy(np.array(labels))
            # if x.shape[0] > 6e5: continue
            # forward
            out_set = self.model(x, training=True)
            # loss    
                
            # mse, mse_list = 0, []
            # for out_cls, ground_truth in zip(out_set['y_b'], out_set['y']):
            #     curr_mse = get_mse(out_cls, ground_truth)
            #     mse += curr_mse
            #     mse_list.append(curr_mse.item()) 


            bits_b = get_bits(out_set['likelihood_b'])
            bits_b_avg = bits_b / len(out_set['nums_list'][2])
          
            ## CE classification
            ce_classification = get_CE_loss(out_set['logits'], labels.to(device)) 

            # sum_loss = self.config.alpha * ce_classification + self.config.gamma * mse + self.config.beta * (bits_b_avg)
            sum_loss = self.config.alpha * ce_classification + self.config.beta * (bits_b_avg)
            
            ## loss for base branch
            # sum_loss = self.config.alpha * ce_classification + self.config.beta * bits_b_avg
            # sum_loss = ce_classification
            # sum_loss = self.config.alpha * mse + self.config.beta * bpp_e

            # # backward & optimize
            # sum_loss.requires_grad = True
            sum_loss.backward()

            # loss_b = self.config.alpha * ce_classification + self.config.beta * bits_b_avg
            # backward & optimize
            # loss_b.requires_grad = True
            # loss_b.backward()
  
            self.optimizer.step()
            # metric & record
            with torch.no_grad():
                # statistics
                logits = out_set['logits']
                preds_class = torch.argmax(logits, 1)
                labels_class = labels.to(device)

                running_corrects_class = torch.sum(preds_class == labels_class)
                

                # self.record_set['mse'].append(mse.item())
                # self.record_set['mses'].append(mse_list)
                self.record_set['ce_cls'].append(ce_classification.item())
                self.record_set['bits_b'].append(bits_b_avg.item())
                self.record_set['sum_loss'].append(sum_loss.item())

                self.accuracy_set['number_correct'].append(running_corrects_class.item())
                self.accuracy_set['total_number'].append(labels.size(0))

            torch.cuda.empty_cache()# empty cache.

        with torch.no_grad(): self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)
        model_path = self.save_model()
        self.epoch += 1

        return model_path
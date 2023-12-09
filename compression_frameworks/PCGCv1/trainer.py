import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch
from dataprocess.inout_points import load_points, save_points, points2voxels, select_voxels

from loss import get_bce_loss, get_classify_metrics
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
       
        self.start = time.time()
        self.model = model
        self.load_state_dict() 
        self.model = torch.nn.DataParallel(self.model) 
        self.model.to(device) 
        self.logger.info(self.model)
        self.epoch = 0
        self.DISPLAY_STEP = 30 #100
        self.SAVE_STEP = 300 #1000
        self.record_set = {'bpp_ae':[], 'bpp_hyper':[], 'bpp':[],'IoU':[]} 
        self.optimizer = self.set_optimizer() 

    def getlogger(self, logdir): 
        logger = logging.getLogger(__name__) 
        logger.setLevel(level = logging.INFO) #NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
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
            if isinstance(self.model, torch.nn.DataParallel): 
                self.model.module.load_state_dict(ckpt['model'])
            else: 
                self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)

        return
    
    def update_lr(self,lr):
        self.config.lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return

    def save_model(self,global_step=None): 
        if global_step == None:
            save_dir = os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth')
        else:
            save_dir = os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '_' + str(int(global_step)) + '.pth')
        state = self.model.module.state_dict() if isinstance(self.model, torch.nn.DataParallel) else self.model.state_dict()
        torch.save({'model': state, 'optimizer':self.optimizer.state_dict()}, save_dir) 
        return

    def set_optimizer(self): 
        '''params_lr_list = []
        for module_name in self.model.module._modules.keys():
            #print("module_name:",module_name)
            #params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        #optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)
        '''
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, betas=(0.9, 0.999), weight_decay=1e-4)
        if self.config.init_ckpt!='':
            ckpt = torch.load(self.config.init_ckpt)
            if 'optimizer' in ckpt.keys():
                optimizer.load_state_dict(ckpt['optimizer'])
                for param_group in optimizer.param_groups: 
                    param_group['lr'] = self.config.lr

        return optimizer

    @torch.no_grad()
    def record(self, main_tag, global_step): 
        # print record
        self.logger.info('='*10+main_tag + ' Epoch ' + str(self.epoch) + ' Step: ' + str(global_step))
      
        for k, v in self.record_set.items(): 
            self.logger.info(k+': '+str(np.round(v, 4).tolist()))   
        # return zero
        for k in self.record_set.keys(): 
            self.record_set[k] = []  

        return 

    @torch.no_grad()
    def test(self, dataloader, main_tag='Test'):
        bpps_ae = 0.
        bpps_hyper = 0.
        IoUs = 0. 
        self.logger.info('Testing Files length:' + str(len(dataloader)))
        for _, points in enumerate(tqdm(dataloader)): 
            # data
            x_np = points2voxels(points,64).astype('float32') 
            x_cpu = torch.from_numpy(x_np).permute(0,4,1,2,3) #(8, 64, 64, 64, 1)->(8, 1, 64, 64, 64)
            x = x_cpu.to(device) 
            # # Forward.
            out_set = self.model(x, training=False) 
            # loss    
            #bce, bce_list = 0, []
            num_points = torch.sum(torch.gt(x,0).float())
            train_bpp_ae = torch.sum(torch.log(out_set['likelihoods'])) / (-np.log(2) * num_points)
            train_bpp_hyper = torch.sum(torch.log(out_set['likelihoods_hyper'])) / (-np.log(2) * num_points)

            points_nums = torch.sum(x_cpu,dim=(1,2,3,4)).int()
            x_tilde = out_set['x_tilde'].cpu().numpy() #(8, 1, 64, 64, 64)
            output = select_voxels(x_tilde, points_nums, 1.0) #(8, 1, 64, 64, 64)
            output = torch.from_numpy(output) #CPU
            _, _, IoU = get_classify_metrics(output, x_cpu)
            bpps_ae = bpps_ae + train_bpp_ae.item() 
            bpps_hyper = bpps_hyper + train_bpp_hyper.item()
            IoUs = IoUs + IoU.item() 
            torch.cuda.empty_cache()# empty cache.

        bpps_ae = bpps_ae / len(dataloader)
        bpps_hyper = bpps_hyper / len(dataloader)
        IoUs = IoUs / len(dataloader)
        # record
        self.record_set['bpp_ae'].append(bpps_ae) 
        self.record_set['bpp_hyper'].append(bpps_hyper)
        self.record_set['bpp'].append(bpps_ae+bpps_hyper)
        self.record_set['IoU'].append(IoUs)
        self.record(main_tag=main_tag, global_step=self.epoch)
        return 

    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # optimizer
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tbeta:' + str(round(self.config.beta,2)))
        self.logger.info('LR:' + str(np.round([params['lr'] for params in self.optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        train_bpp_ae_sum = 0.
        train_bpp_hyper_sum = 0.
        train_IoU_sum = 0.
        num = 0.

        start_time = time.time()
        for batch_step, points in enumerate(tqdm(dataloader)): 
            # data
            x_np = points2voxels(points,64).astype('float32') 
            x_cpu = torch.from_numpy(x_np).permute(0,4,1,2,3) # (8, 64, 64, 64, 1)->(8, 1, 64, 64, 64)
            x = x_cpu.to(device) #转到GPU
            # forward
            out_set = self.model(x, training=True)
            # loss    
            num_points = torch.sum(torch.gt(x,0).float()) 
            train_bpp_ae = torch.sum(torch.log(out_set['likelihoods'])) / (-np.log(2) * num_points)
            train_bpp_hyper = torch.sum(torch.log(out_set['likelihoods_hyper'])) / (-np.log(2) * num_points)
            train_zeros, train_ones = get_bce_loss(out_set['x_tilde'], x)
            train_distortion = self.config.beta * train_zeros + 1.0 * train_ones
            train_loss = self.config.alpha * train_distortion + self.config.delta * train_bpp_ae + self.config.gamma * train_bpp_hyper
            train_loss *= 3 
            # backward & optimize
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            if (batch_step + 1) % self.DISPLAY_STEP == 0: 
                self.logger.info('train_zeros:' + str(train_zeros.item()))
                self.logger.info('train_ones:' + str(train_ones.item()))
                self.logger.info('train_distortion:' + str(train_distortion.item()))
                self.logger.info('train_loss:' + str(train_loss.item()))
            del train_loss
            with torch.no_grad():
                # post-process: classification.
                points_nums = torch.sum(x_cpu,dim=(1,2,3,4)).int()
                x_tilde = out_set['x_tilde'].cpu().numpy() #(8, 1, 64, 64, 64)
                output = select_voxels(x_tilde, points_nums, 1.0) #(8, 1, 64, 64, 64)
                output = torch.from_numpy(output) #CPU
                _, _, IoU = get_classify_metrics(output, x_cpu)
                
                train_bpp_ae_sum += train_bpp_ae.item() 
                train_bpp_hyper_sum += train_bpp_hyper.item()
                train_IoU_sum += IoU.item()
                num += 1
                
                # Display
                if (batch_step + 1) % self.DISPLAY_STEP == 0: 
                    train_bpp_ae_sum /= num
                    train_bpp_hyper_sum /= num
                    train_IoU_sum  /= num

                    print("Iteration:{0:}".format(batch_step))
                    print("Bpps: {0:.4f} + {1:.4f}".format(train_bpp_ae_sum, train_bpp_hyper_sum))
                    print("IoU: ", train_IoU_sum)
                    print('Running time:(mins):', round((time.time()-self.start)/60.))
                    print()
                    # record
                    self.record_set['bpp_ae'].append(train_bpp_ae_sum)
                    self.record_set['bpp_hyper'].append(train_bpp_hyper_sum)
                    self.record_set['bpp'].append(train_bpp_ae_sum+train_bpp_hyper_sum)
                    self.record_set['IoU'].append(train_IoU_sum)
                    self.record(main_tag='Train', global_step=self.epoch*len(dataloader)+batch_step)

                    num = 0.
                    train_bpp_ae_sum = 0.
                    train_bpp_hyper_sum = 0.
                    train_IoU_sum = 0.

         
        self.logger.info('Epoch:' + str(self.epoch) + " ,save model!")
        self.save_model(global_step=self.epoch*len(dataloader)+batch_step)
        torch.cuda.empty_cache()
        self.epoch += 1

        return
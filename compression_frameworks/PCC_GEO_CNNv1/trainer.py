import os, time, logging
from tqdm import tqdm
import numpy as np
import torch
from inout_points import points2voxels

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def focal_loss(y_true, y_pred, gamma=2, alpha=0.95):
    # pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_1 = torch.where(torch.eq(y_true,1), y_pred, torch.ones_like(y_pred))

    # pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))

    # pt_1 = K.clip(pt_1, 1e-3, .999)
    pt_1 = torch.clamp(pt_1, min=1e-3, max=.999)
    # pt_0 = K.clip(pt_0, 1e-3, .999)
    pt_0 = torch.clamp(pt_0, min=1e-3, max=.999)

    # return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return -torch.sum(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - torch.sum((1-alpha) * torch.pow( pt_0, gamma) * torch.log(1. - pt_0))

class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.start = time.time()
        self.model = model.to(device)
        self.logger.info(model)
        self.best_loss = None
        self.best_quantiles_loss = None
        self.load_state_dict()
        self.epoch = 0
        self.DISPLAY_STEP = 20 #
        self.SAVE_STEP = 20 #1000
        self.resolution = config.resolution
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.lmbda = config.lmbda
        self.main_optimizer = None
        self.aux_optimizer = None
        self.main_optimizer,self.aux_optimizer = self.set_optimizer()

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
            self.model.load_state_dict(ckpt['model'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)
            self.best_loss = ckpt['best_loss']
            if self.best_loss < 3.5:
                self.best_loss = 3.5
            self.logger.info('best_loss: ' + str(self.best_loss))
            self.best_quantiles_loss = ckpt['best_quantiles_loss']
            if self.best_quantiles_loss < 10:
                self.best_quantiles_loss = 10
            self.logger.info('best_quantiles_loss: ' + str(self.best_quantiles_loss))
        return

    def save_model(self,global_step=None): 
        if global_step == None:
            save_dir = os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth')
        else:
            save_dir = os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '_' + str(int(global_step)) + '.pth')
        torch.save({'model': self.model.state_dict(),'best_loss':self.best_loss,'best_quantiles_loss':self.best_quantiles_loss}, save_dir)
        return

    def set_optimizer(self): 
        params_lr_list = []
        for module_name in self.model._modules.keys():
            print("main module_name:",module_name)
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.lr})
        main_optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)
        if self.main_optimizer != None:
            main_optimizer.load_state_dict(self.main_optimizer.state_dict())
            for param_group in main_optimizer.param_groups: 
                param_group['lr'] = self.config.lr

        aux_params_lr_list = []
        for pname, p in self.model._modules['entropy_bottleneck'].named_parameters():
            if pname == 'quantiles':
                aux_params_lr_list += [p]

        aux_optimizer = torch.optim.Adam([{"params":aux_params_lr_list, 'lr':self.config.aux_lr}], betas=(0.9, 0.999), weight_decay=1e-4)
       
        if self.aux_optimizer != None:
            aux_optimizer.load_state_dict(self.aux_optimizer.state_dict())
            for param_group in aux_optimizer.param_groups: 
                param_group['lr'] = self.config.aux_lr

        return main_optimizer,aux_optimizer

    @torch.no_grad()
    def test(self, dataloader):
        self.logger.info('Epoch:'+str(self.epoch)+', Testing Files length:' + str(len(dataloader)))
        total_loss = 0.
        total_fl = 0.
        total_mbpov = 0.
        total_quantiles_loss = 0.
        for _, points in enumerate(tqdm(dataloader)): 
            # data
            x_np = points2voxels(points,self.resolution).astype('float32') 
            x_cpu = torch.from_numpy(x_np).permute(0,4,1,2,3)  #(8, 64, 64, 64, 1)->(8, 1, 64, 64, 64)
            x = x_cpu.to(device) 
            # # Forward.
            out_set = self.model(x, training=False) 
            # loss    
            num_points = torch.sum(torch.gt(x,0).float()) 
            test_mbpov = torch.sum(torch.log(out_set['likelihoods'])) / (-np.log(2) * num_points)

            test_fl = focal_loss(x, out_set['x_tilde'], gamma=self.gamma, alpha=self.alpha)
            test_loss = self.lmbda * test_fl + test_mbpov
            total_loss += test_loss.item()
            total_fl += test_fl.item()
            total_mbpov += test_mbpov.item()
            total_quantiles_loss += out_set['quantiles_loss'].item()
            torch.cuda.empty_cache()# empty cache.

        total_loss = total_loss / len(dataloader)
        total_fl = total_fl / len(dataloader)
        total_mbpov = total_mbpov / len(dataloader)
        total_quantiles_loss = total_quantiles_loss / len(dataloader)
        # record
        self.logger.info('total_loss:' + str(total_loss))
        self.logger.info('total_fl:' + str(total_fl))
        self.logger.info('total_mbpov:' + str(total_mbpov))
        self.logger.info('total_quantiles_loss:' + str(total_quantiles_loss))
        return 

    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        # main_optimizer
        # self.main_optimizer = self.set_optimizer()
        self.logger.info('alpha:' + str(round(self.alpha,4)) + '\tgamma:' + str(round(self.gamma,4)) + '\tlmbda:' + str(round(self.lmbda,6)))
        self.logger.info('main LR:' + str(np.round([params['lr'] for params in self.main_optimizer.param_groups], 6).tolist()))
        self.logger.info('aux LR:' + str(np.round([params['lr'] for params in self.aux_optimizer.param_groups], 6).tolist()))
        # dataloader
        self.logger.info('Training Files length:' + str(len(dataloader)))
        total_loss = 0.
        total_fl = 0.
        total_mbpov = 0.
        total_quantiles_loss = 0.
        num = 0.

        for batch_step, points in enumerate(tqdm(dataloader)): 
            self.main_optimizer.zero_grad()
            self.aux_optimizer.zero_grad()
            # data
            x_np = points2voxels(points,self.resolution).astype('float32') 
            x_cpu = torch.from_numpy(x_np).permute(0,4,1,2,3) # (8, 64, 64, 64, 1)->(8, 1, 64, 64, 64)
            x = x_cpu.to(device) 
            # forward
            out_set = self.model(x, training=True)
            # loss    
            num_points = torch.sum(torch.gt(x,0).float()) 
            train_mbpov = torch.sum(torch.log(out_set['likelihoods'])) / (-np.log(2) * num_points)

            train_fl = focal_loss(x, out_set['x_tilde'], gamma=self.gamma, alpha=self.alpha)
            train_loss = self.lmbda * train_fl + train_mbpov

            # backward & optimize
            train_loss.backward()
            self.main_optimizer.step()
            out_set['quantiles_loss'].backward()
            self.aux_optimizer.step()

            with torch.no_grad():
                
                total_loss += train_loss.item()
                total_fl += train_fl.item()
                total_mbpov += train_mbpov.item()
                total_quantiles_loss += out_set['quantiles_loss'].item()
                num += 1
                
                # Display
                if (batch_step + 1) % self.DISPLAY_STEP == 0 or (batch_step + 1) % self.SAVE_STEP == 0: 
                    total_loss /= num
                    total_fl /= num
                    total_mbpov  /= num
                    total_quantiles_loss  /= num

                    # Save checkpoints.
                    if (batch_step + 1) % self.SAVE_STEP == 0:
                        if self.best_loss is None or self.best_loss > total_loss or self.best_quantiles_loss > total_quantiles_loss:
                            if self.best_loss > total_loss:
                                self.best_loss = total_loss
                            if self.best_quantiles_loss > total_quantiles_loss:
                                self.best_quantiles_loss = total_quantiles_loss
                            self.logger.info('Iteration:' + str(self.epoch*len(dataloader)+batch_step) + " ,save model!")
                            self.save_model(global_step=self.epoch*len(dataloader)+batch_step)
                    # record
                    self.logger.info('Training step:' + str(self.epoch*len(dataloader)+batch_step))
                    self.logger.info('total_loss:' + str(total_loss))
                    self.logger.info('total_fl:' + str(total_fl))
                    self.logger.info('total_mbpov:' + str(total_mbpov))
                    self.logger.info('total_quantiles_loss:' + str(total_quantiles_loss))

                    num = 0.
                    total_loss = 0.
                    total_fl = 0.
                    total_mbpov = 0.
                    total_quantiles_loss = 0.

            torch.cuda.empty_cache()# empty cache.
        self.epoch += 1

        return
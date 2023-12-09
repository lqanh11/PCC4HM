import os, sys, time, logging
from tqdm import tqdm
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def focal_loss(y_true, y_pred, gamma=2, alpha=0.9):
    pt_1 = torch.where(torch.eq(y_true,1), y_pred, torch.ones_like(y_pred))
    pt_0 = torch.where(torch.eq(y_true, 0), y_pred, torch.zeros_like(y_pred))

    pt_1 = torch.clamp(pt_1, min=1e-3, max=.999)
    pt_0 = torch.clamp(pt_0, min=1e-3, max=.999)

    return -torch.sum(alpha * torch.pow(1. - pt_1, gamma) * torch.log(pt_1)) - torch.sum((1-alpha) * torch.pow( pt_0, gamma) * torch.log(1. - pt_0))

class Trainer():
    def __init__(self, config, model):
        self.config = config
        self.logger = self.getlogger(config.logdir)
        self.start = time.time()
        self.model = model.to(device)
        self.logger.info(model)
        self.main_optimizer = None
        self.aux_optimizer = None
        self.load_state_dict()
        self.epoch = 0
        self.DISPLAY_STEP = 10 #1个step 2s
        self.SAVE_STEP = self.DISPLAY_STEP*2 #1000
        self.resolution = config.resolution
        self.best_loss = None
        self.best_quantiles_loss = None
        self.best_mbpov_loss = None
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.lmbda = config.lmbda
        self.set_optimizer()

    def getlogger(self, logdir): #不用改
        logger = logging.getLogger(__name__) #提供了应用程序可以直接使用的接口
        logger.setLevel(level = logging.INFO) #NOTSET < DEBUG < INFO < WARNING < ERROR < CRITICAL
        handler = logging.FileHandler(os.path.join(logdir, 'log.txt')) #将(logger创建的)日志记录发送到合适的目的输出
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S') #决定日志记录的最终输出格式
        handler.setFormatter(formatter)
        console = logging.StreamHandler() #用于输出到控制台
        console.setLevel(logging.INFO)
        console.setFormatter(formatter)
        logger.addHandler(handler)
        logger.addHandler(console)

        return logger

    def load_state_dict(self): #不用改
        """selectively load model
        """
        params_lr_list = []
        for module_name in self.model._modules.keys():
            print("module_name:",module_name) #这里要打印看看都有哪些参数会被训练
            if module_name == 'entropy_bottleneck':
                for pname, p in self.model._modules[module_name].named_parameters():
                    print(pname)
                    # if pname == 'quantiles':
                    #     print(p)
            params_lr_list.append({"params":self.model._modules[module_name].parameters(), 'lr':self.config.main_lr}) #也包含quantiles参数
        self.main_optimizer = torch.optim.Adam(params_lr_list, betas=(0.9, 0.999), weight_decay=1e-4)
        aux_params_lr_list = []
        for pname, p in self.model._modules['entropy_bottleneck'].named_parameters():
            if pname == 'quantiles':
                aux_params_lr_list += [p]
        # print("aux_params_lr_list:",aux_params_lr_list)
        self.aux_optimizer = torch.optim.Adam([{"params":aux_params_lr_list, 'lr':self.config.aux_lr}], betas=(0.9, 0.999), weight_decay=1e-4)
        if self.config.init_ckpt=='':
            self.logger.info('Random initialization.')
        else:
            ckpt = torch.load(self.config.init_ckpt)
            self.model.load_state_dict(ckpt['model'])
            self.main_optimizer.load_state_dict(ckpt['main_optimizer'])
            self.aux_optimizer.load_state_dict(ckpt['aux_optimizer'])
            self.logger.info('Load checkpoint from ' + self.config.init_ckpt)
            self.best_loss = ckpt['best_loss']
            # if self.best_loss < 3.5:
            #     self.best_loss = 3.5
            self.logger.info('best_loss: ' + str(self.best_loss))
            self.best_quantiles_loss = ckpt['best_quantiles_loss']
            if 'best_mbpov_loss' in ckpt.keys():
                self.best_mbpov_loss = ckpt['best_mbpov_loss']
                self.logger.info('best_mbpov_loss: ' + str(self.best_mbpov_loss))
            # if self.best_quantiles_loss < 10:
            #     self.best_quantiles_loss = 10
            self.logger.info('best_quantiles_loss: ' + str(self.best_quantiles_loss))
        return

    def save_model(self,global_step=None): #已改
        if global_step == None:
            save_dir = os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '.pth')
        else:
            save_dir = os.path.join(self.config.ckptdir, 'epoch_' + str(self.epoch) + '_' + str(int(global_step)) + '.pth')
        torch.save({'model': self.model.state_dict(),'main_optimizer':self.main_optimizer.state_dict(),
            'aux_optimizer':self.aux_optimizer.state_dict(),'best_loss':self.best_loss,
            'best_quantiles_loss':self.best_quantiles_loss,'best_mbpov_loss':self.best_mbpov_loss}, save_dir)
        return

    def set_optimizer(self): #已改
        for param_group in self.main_optimizer.param_groups: #修改lr
            param_group['lr'] = self.config.main_lr
        for param_group in self.aux_optimizer.param_groups: #修改lr
            param_group['lr'] = self.config.aux_lr

    #以下test已改
    @torch.no_grad()
    def test(self, dataloader):
        self.logger.info('Epoch:'+str(self.epoch)+', Testing Files length:' + str(len(dataloader)))
        total_loss = 0.
        total_fl = 0.
        total_mbpov = 0.
        total_quantiles_loss = 0.
        for _, points in enumerate(tqdm(dataloader)): #points是单纯点的集合
            # data
            x_cpu = torch.from_numpy(np.array(points)) #(32, 1, 64, 64, 64)
            # print("x_cpu.shape:",x_cpu.shape)
            x = x_cpu.to(device) #转到GPU
            # # Forward.
            out_set = self.model(x, training=False) #在GPU
            # loss    
            num_occupied_voxels = torch.sum(x) #x各维度点的总个数
            log_y_likelihoods = torch.log(out_set['y_likelihoods'])
            log_z_likelihoods = torch.log(out_set['z_likelihoods'])
            # loss    
            denominator = -np.log(2) * num_occupied_voxels
            test_mbpov_y = torch.sum(log_y_likelihoods) / denominator
            test_mbpov_z = torch.sum(log_z_likelihoods) / denominator
            test_mbpov = test_mbpov_y + test_mbpov_z
            test_fl = focal_loss(x, out_set['x_tilde'], gamma=self.config.gamma, alpha=self.config.alpha)
            test_loss = self.config.lmbda * test_fl + test_mbpov
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

    #已改train
    def train(self, dataloader):
        self.logger.info('='*40+'\n'+'Training Epoch: ' + str(self.epoch))
        self.logger.info('alpha:' + str(round(self.config.alpha,2)) + '\tgamma:' + str(round(self.config.gamma,2))
             + '\tlmbda:' + str(round(self.config.lmbda,2)))
        self.logger.info('main LR:' + str(np.round([params['lr'] for params in self.main_optimizer.param_groups], 6).tolist()))
        self.logger.info('aux LR:' + str(np.round([params['lr'] for params in self.aux_optimizer.param_groups], 6).tolist()))
        self.logger.info('Training Files length:' + str(len(dataloader)))
        total_loss = 0.
        total_fl = 0.
        total_mbpov = 0.
        total_quantiles_loss = 0.
        num = 0.
        for batch_step, points in enumerate(tqdm(dataloader)): #循环一次一个batch，points是单纯点的集合
            self.main_optimizer.zero_grad()
            self.aux_optimizer.zero_grad()

            x_cpu = torch.from_numpy(np.array(points)) #(32, 1, 64, 64, 64)
            # print("x_cpu.shape:",x_cpu.shape) #x_cpu.shape: torch.Size([32, 1, 64, 64, 64])
            x = x_cpu.to(device) #转到GPU
            # forward
            out_set = self.model(x, training=True)
            num_occupied_voxels = torch.sum(x) #x各维度点的总个数
            log_y_likelihoods = torch.log(out_set['y_likelihoods'])
            log_z_likelihoods = torch.log(out_set['z_likelihoods'])
            # loss    
            denominator = -np.log(2) * num_occupied_voxels
            train_mbpov_y = torch.sum(log_y_likelihoods) / denominator
            train_mbpov_z = torch.sum(log_z_likelihoods) / denominator
            train_mbpov = train_mbpov_y + train_mbpov_z
            train_fl = focal_loss(x, out_set['x_tilde'], gamma=self.config.gamma, alpha=self.config.alpha)
            train_loss = self.config.lmbda * train_fl + train_mbpov

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
                if (batch_step + 1) % self.DISPLAY_STEP == 0 or (batch_step + 1) % self.SAVE_STEP == 0: #每100 step record一次
                    total_loss /= num
                    total_fl /= num
                    total_mbpov  /= num
                    total_quantiles_loss  /= num

                    # Save checkpoints.
                    if (batch_step + 1) % self.SAVE_STEP == 0:
                        if self.best_loss is None or self.best_loss > total_loss or self.best_mbpov_loss is None or self.best_mbpov_loss > total_mbpov: #self.best_quantiles_loss > total_quantiles_loss:
                            if self.best_loss is None or self.best_loss > total_loss:
                                self.best_loss = total_loss
                            # if self.best_quantiles_loss is None or self.best_quantiles_loss > total_quantiles_loss:
                            #     self.best_quantiles_loss = total_quantiles_loss
                            if self.best_mbpov_loss is None or self.best_mbpov_loss > total_mbpov:
                                self.best_mbpov_loss = total_mbpov
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
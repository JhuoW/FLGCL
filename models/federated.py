import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from datahelper.loader import DataLoader
import os.path as osp
from misc.utils import *
import numpy as np
from misc.logger import Logger

class ClientModule:
    def __init__(self, args, config, work_id, gpu_id, sd):
        self.sd = sd
        self.gpu_id = gpu_id
        self.worker_id = work_id
        self.args = args 
        self._args = vars(self.args)
        self.loader = DataLoader(self.args)
        self.config = config
        self.logger = Logger(args=self.args, gpu_id = self.gpu_id, is_server = False)
       
    def switch_state(self, client_id):  
        self.client_id = client_id
        self.loader.switch(client_id)  # 切换到这个client，从文件中读取这个client的子图
        self.logger.switch(client_id)  # 切换到当前这个client
        if self.is_initialized():
            time.sleep(0.1)
            self.load_state()
        else:
            self.init_state()

    def is_initialized(self):
        return osp.exists(osp.join(self.args.checkpt_path, f'{self.client_id}_state.pt'))

    @property
    def init_state(self):
        raise NotImplementedError()

    @property
    def save_state(self):
        raise NotImplementedError()

    @property
    def load_state(self):
        raise NotImplementedError()

    @torch.no_grad()
    def validate(self, mode='test'):
        loader = self.loader.patition_loader  # client loader， 每次加载一个client

        with torch.no_grad():
            target, pred, loss = [], [], []
            for _, batch in enumerate(loader):  # 加载一个client的subgraph
                batch = batch.cuda(self.gpu_id)  
                mask = batch.test_mask if mode == 'test' else batch.val_mask
                y_hat, lss = self.validation_step(batch, mask)  
                pred.append(y_hat[mask])  # 预测logits
                target.append(batch.y[mask])  # 真实label
                loss.append(lss)
            # 预测logits, 真实stacked labels
            acc = self.accuracy(torch.stack(pred).view(-1, self.config['num_cls']), torch.stack(target).view(-1))
        return acc, np.mean(loss)



    @torch.no_grad()
    def validation_step(self, batch, mask=None):
        self.local_model.eval()
        y_hat = self.local_model(batch)
        if torch.sum(mask).item() == 0: return y_hat, 0.0
        lss = self.loss_func(y_hat[mask], batch.y[mask])
        return y_hat, lss.item()

    @torch.no_grad()
    def accuracy(self, preds, targets):
        if targets.size(0) == 0: return 1.0
        with torch.no_grad():
            preds = preds.max(1)[1] # 每行的最大值id
            acc = preds.eq(targets).sum().item() / targets.size(0)
        return acc

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

    def save_log(self):
        save(self.args.log_path, f'client_{self.client_id}.txt', {
            'args': self._args,
            'log': self.log,
            'config': self.config
        })

    def get_optimizer_state(self, optimizer):
        state = {}
        for param_key, param_values in optimizer.state_dict()['state'].items():
            state[param_key] = {}
            for name, value in param_values.items():
                if torch.is_tensor(value) == False: continue
                state[param_key][name] = value.clone().detach().cpu().numpy()
        return state


class ServerModule:
    def __init__(self, args, config, sd, gpu_server):
        self.args = args
        self._args = vars(self.args)
        self.config = config
        self.sd = sd
        self.gpu_id = gpu_server
        self.logger = Logger(args=self.args, gpu_id = self.gpu_id, is_server = True)
    

    def aggregate(self, local_weights, ratio=None):
        # ratio = [1/10, 1/10, 1/10,..., 1/10]
        # local_weights: [weights of client1, weights of client2, ...]
        # 用于得到client端的训练参数名
        aggr_theta = OrderedDict([(k, None) for k in local_weights[0].keys()])   # 第一个client的参数模型的key（参数名）

        if ratio is not None:
            for name, param in aggr_theta.items(): # 遍历所有参数名
                # 遍历每个client的参数，得到client id j和参数theta
                # theta[name]表示每个client的名称为name的参数
                # 对所有client的name 参数做的值做加权求和，赋值个aggr_theta的name下
                aggr_theta[name] = np.sum([theta[name] * ratio[j] for j, theta in enumerate(local_weights)], 0)
        else:
            ratio = 1 / len(local_weights)
            for name, param in aggr_theta.items():
                aggr_theta[name] = aggr_theta[name] = np.sum([theta[name] * ratio for _, theta in enumerate(local_weights)], 0)
        return aggr_theta
from models.federated import ClientModule
from models.nets import GRACE
import torch
import time

class Client(ClientModule):
    def __init__(self, args, config, work_id, gpu_id, sd):
        super(Client, self).__init__(args, work_id, gpu_id, sd)
        self.local_model = GRACE(config).cuda()
        self.parameters = list(self.local_model.parameters())
        self.config = config

    def init_state(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr = self.config['local_lr'], weight_decay=self.config['local_wd'])
        self.log = {'lr': [],'train_lss': [],
                    'ep_local_val_acc_avg': [], 'ep_local_val_acc_std': [],
                    'ep_local_val_f1ma_avg': [], 'ep_local_val_f1ma_std': [],
                    'rnd_local_val_acc_avg': [], 'rnd_local_val_acc_std': [],
                    'rnd_local_val_f1ma_avg': [], 'rnd_local_val_f1ma_std': [],

                    'ep_local_test_acc_avg': [], 'ep_local_test_acc_std': [],
                    'ep_local_test_f1ma_avg': [], 'ep_local_test_f1ma_std': [],
                    'rnd_local_test_acc_avg': [], 'rnd_local_test_acc_std': [],
                    'rnd_local_test_f1ma_avg': [], 'rnd_local_test_f1ma_std': []}    

    def save_state(self):
        return


    def train_client(self):
        st = time.time()
        val_local_results = self.eval_LR(mode='val')
        test_local_results = self.eval_LR(mode='test')
        self.logger.print_fl(  # 打印当前client的训练信息
            f'rnd: {self.curr_rnd+1}, ep: {0}, '
            f'local_val_acc = {val_local_results['acc_avg']:.4f} +- {val_local_results['acc_std']:.4f}, '
            f'local_val_f1ma = {val_local_results['f1ma_avg']:.4f} +- {val_local_results['f1ma_std']:.4f},'
            f'lr: {self.get_lr()} ({time.time()-st:.2f}s)'
        )
        self.log['ep_local_val_acc_avg'].append(val_local_results['acc_avg'])
        self.log['ep_local_val_acc_std'].append(val_local_results['acc_std'])
        self.log['ep_local_val_f1ma_avg'].append(val_local_results['f1ma_avg'])       
        self.log['ep_local_val_f1ma_std'].append(val_local_results['f1ma_std'])  

        self.log['ep_local_test_acc_avg'].append(test_local_results['acc_avg'])
        self.log['ep_local_test_acc_std'].append(test_local_results['acc_std'])
        self.log['ep_local_test_f1ma_avg'].append(test_local_results['f1ma_avg'])       
        self.log['ep_local_test_f1ma_std'].append(test_local_results['f1ma_std'])  

        for ep in range(self.config['local_epochs']):
            st = time.time()
            self.local_model.train()
            for _, batch in enumerate(self.loader.patition_loader):
                self.optimizer.zero_grad()
                batch = batch.cuda()
                
                
        return
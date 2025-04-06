from models.federated import ClientModule
from models.nets import NLGNN, NLGNN2, GCN
from models.fedpubnets import MaskedGCN
import torch
import torch.nn as nn
from misc.utils import *

class Client(ClientModule):
    def __init__(self, args, config, work_id, gpu_id, sd):
        '''
        work_id: 处理当前client的线程id
        gpu_id: 当前client使用的gpu id
        sd: 当前client中和server shared data (对于FedAvg来说是模型参数和模型size)
        '''
        super(Client, self).__init__(args, config, work_id, gpu_id, sd)
        self.local_model = NLGNN(config).cuda(gpu_id)
        # self.local_model = NLGNN2(n_gnn_layers=config['n_gnn_layers'], 
        #                           in_dim=config['num_feats'], 
        #                           hid_dim=config['gnn_hid_dim'], 
        #                           out_dim=config['num_cls'],
        #                           kernel=config['kernel'],
        #                           dropout1=config['dropout1'],
        #                           dropout2=config['dropout2']).cuda(gpu_id)
        # self.local_model = GCN(n_gnn_layers=config['n_gnn_layers'], 
        #                        in_dim=config['num_feats'], 
        #                        hid_dim=config['gnn_hid_dim'], 
        #                        out_dim=config['num_cls'],
        #                        kernel=config['kernel'],
        #                        dropout1=config['dropout1'],
        #                        dropout2=config['dropout2']).cuda(gpu_id)
        # self.local_model = MaskedGCN(config['num_feats'], 128, config['num_cls'], 0.001, self.args).cuda(gpu_id)
        self.parameters = list(self.local_model.parameters())

    def init_state(self):
        self.optimizer = torch.optim.Adam(self.parameters, lr = self.config['local_lr'], weight_decay=self.config['local_wd'])
        self.loss_func = nn.CrossEntropyLoss()
        self.log = {'lr': [],'train_lss': [],
                    'ep_local_val_acc': [], 'ep_local_val_loss': [],
                    'rnd_local_val_acc': [], 'rnd_local_val_loss': [],

                    'ep_local_test_acc': [], 'ep_local_test_loss': [],
                    'rnd_local_test_acc': [], 'rnd_local_test_loss': []}    
        

    def load_state(self):
        state = torch_load(self.args.checkpt_path, f'{self.client_id}_state.pt')
        set_state_dict(self.local_model, state['local_model'], self.gpu_id) # 加载模型参数
        self.optimizer.load_state_dict(state['optimizer'])  # 加载优化器参数
        self.log = state['log']  # 加载日志        

    def save_state(self):
        torch_save(self.args.checkpt_path, f'{self.client_id}_state.pt', {
            'local_model': get_state_dict(self.local_model),
            'optimizer': self.optimizer.state_dict(),
            'log': self.log
        })

    def on_receive_message(self, curr_comm):
        self.curr_comm = curr_comm
        # fedavg
        self.update(self.sd['global']) # 传入global server的信息 并将global的模型信息拷贝给local模型

    def update(self, received):
        # received['model']是global server的模型参数
        set_state_dict(self.local_model, received['model'], self.gpu_id, skip_stat = True) # 将server端的global参数赋值给client

    def on_communication_begin(self):
        self.train_client()
        self.transfer_to_server()


    def train_client(self):
        '''
        进行当前client的一次communication的训练
        '''
        # val_local_results = self.eval_LR(mode='val')
        # test_local_results = self.eval_LR(mode='test')
        val_local_acc, val_local_loss = self.validate(mode = 'val')
        test_local_acc, test_local_loss = self.validate(mode = 'test')        
        self.logger.print_fl(    # 打印当前client的信息
            f'Communication: {self.curr_comm+1}, ep: {0}, '
            f'val_local_loss: {val_local_loss:.4f}, val_local_acc: {val_local_acc:.4f}, test_local_acc: {test_local_acc:.4f}, lr: {self.get_lr()} '
        )
        self.log['ep_local_val_acc'].append(val_local_acc)
        self.log['ep_local_val_loss'].append(val_local_loss)
        self.log['ep_local_test_acc'].append(test_local_acc)
        self.log['ep_local_test_loss'].append(test_local_loss)

        self.local_model.reset_parameters()

        for ep in range(self.config['local_epochs']):
            self.local_model.train()
            for _, batch in enumerate(self.loader.patition_loader):
                self.optimizer.zero_grad()
                batch = batch.cuda(self.gpu_id)  # 一个batch是一个client的子图
                logits = self.local_model(batch)  # logits
                loss = self.loss_func(logits[batch.train_mask], batch.y[batch.train_mask]) 
                loss.backward()
                self.optimizer.step()
            val_local_acc, val_local_loss = self.validate(mode = 'val')
            test_local_acc, test_local_loss = self.validate(mode = 'test')
            self.logger.print_fl(f'Communication: {self.curr_comm+1}, ep: {ep+1}, '
                                 f'val_local_loss: {val_local_loss:.4f}, val_local_acc: {val_local_acc:.4f},'
                                 f'test_local_acc: {test_local_acc:.4f}, lr: {self.get_lr()}')
            self.log['train_lss'].append(loss.item())
            self.log['ep_local_val_acc'].append(val_local_acc)
            self.log['ep_local_val_loss'].append(val_local_loss)
            self.log['ep_local_test_acc'].append(test_local_acc)
            self.log['ep_local_test_loss'].append(test_local_loss)

        self.log['rnd_local_val_acc'].append(val_local_acc)
        self.log['rnd_local_val_loss'].append(val_local_loss)
        self.log['rnd_local_test_acc'].append(test_local_acc)
        self.log['rnd_local_test_loss'].append(test_local_loss)
        self.save_log()

                

    def transfer_to_server(self): # 发送到server的数据  sd(shared data): 用于存放client和server可以共享的数据
        self.sd[self.client_id] = {
            'local_model': get_state_dict(self.local_model),
            'client_size': len(self.loader.partition), 
        }

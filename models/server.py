from models.federated import ServerModule
from models.nets import NLGNN, GCN
import time
from misc.utils import *


class Server(ServerModule):
    def __init__(self, args, config, sd, gpu_server):
        '''
        sd: 存放所有clients的模型数据和server的模型数据
        '''
        super(Server, self).__init__(args, config, sd, gpu_server)
        self.global_model = NLGNN(config).cuda(self.gpu_id)
        # self.global_model = GCN(n_gnn_layers=config['n_gnn_layers'], 
        #                        in_dim=config['num_feats'], 
        #                        hid_dim=config['gnn_hid_dim'], 
        #                        out_dim=config['num_cls'],
        #                        kernel=config['kernel'],
        #                        dropout1=config['dropout1'],
        #                        dropout2=config['dropout2']).cuda(self.gpu_id)
        self.update_lists = []


    def on_communication_begin(self, curr_comm):
        self.communication_begin = time.time()
        self.curr_comm = curr_comm
        self.sd['global'] = self.get_global_weights()  # 将server的模型数据加载到sd中

    def on_communication_end(self, updated):  
        '''
        updated: 存放一次communication更新过的client id
        '''
        self.update(updated)
        self.save_state()
    
    def update(self, updated):
        '''
        updatad: 一次communication更新过的client ID
        该方法聚合updated中的client的参数 通过self.aggregate方法聚合，然后赋值给self.global_model
        '''
        st = time.time()
        local_weights = [] # 存放client上传的数据
        local_client_sizes = []
        # update all clients to server
        for client_id in updated:
            local_weights.append(self.sd[client_id]['local_model'].copy())  # 从client上接受到的模型参数数据
            local_client_sizes.append(self.sd[client_id]['client_size'])  # [1,1,1,1,1,1,1,1,1,1]
            del self.sd[client_id]  # 从sd上删除该client
        self.logger.print_fl('All Clients have been uploaded to server')
        assert len(local_weights) == self.args.n_clients, f'ERROR: len(local_weights)={len(local_weights)} != {self.args.n_clients}'
        num_clients = round(self.args.n_clients * self.args.frac)  # 选择一部分client进行更新

        ratio = (np.array(local_client_sizes)/np.sum(local_client_sizes)).tolist()  #  [1/10, 1/10, 1/10,..., 1/10]
        # 将clients上的参数依照参数名做加权求和后赋值给server端的global model
        self.set_weights(self.global_model,self.aggregate(local_weights, ratio))  
        self.logger.print_fl(f'Global model updated in the round: {self.curr_comm}')

        
    

    def set_weights(self, model, state_dict):
        set_state_dict(model, state_dict, self.gpu_id)


    def get_global_weights(self):
        return {
            'model': get_state_dict(self.global_model),
        }

    def save_state(self):
        torch.save(self.args.checkpt_path, 'server_state.pt', {
            'global_model': get_state_dict(self.global_model),

        })
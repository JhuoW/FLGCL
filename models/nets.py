import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from importlib import import_module
import torch_geometric.nn as pyg_nn

class GCN(nn.Module):
    def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
        super().__init__()
        self.n_feat = n_feat
        self.n_dims = n_dims
        self.n_clss = n_clss
        self.args = args

        self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
        self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
        self.clsif = nn.Linear(self.n_dims, self.n_clss)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.conv1(x, edge_index, edge_weight))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.clsif(x)
        return x
    
class Encoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int, out_dim: int, activation, BaseModel=GCNConv, k: int = 2) :
        super(Encoder, self).__init__()
        self.k = k
        self.conv_list = nn.ModuleList()
        # self.conv_list.append(BaseModel(in_channels, 2 * out_channels))
        for i in range(k):  # 0, 1
            if i == k-1:
                hid_dim = out_dim
            self.conv_list.append(BaseModel(in_dim, hid_dim))
            in_dim = hid_dim
        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv_list[i](x, edge_index))
        return x

class Augmentator(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

class GRACE(nn.Module):
    def __init__(self, config):
        super(GRACE, self).__init__()
        self.config = config
        gnn_hid_dim = config['gnn_hid_dim']
        gnn_out_dim = config['gnn_out_dim']
        gnn_act = ({'relu': F.relu, 'prelu': nn.PReLU(), 'elu': F.elu})[config['gnn_act']]
        BaseGNN = getattr(pyg_nn, config.get('BaseGNN', 'GCNConv'))
        n_gnn_layers = config['n_gnn_layers']
        num_proj_hidden = config['num_proj_hidden']
        in_dim  = config['num_feats']
        self.encoder = Encoder(in_dim, gnn_hid_dim, gnn_out_dim, gnn_act, BaseGNN, k = n_gnn_layers)

        self.fc1 = torch.nn.Linear(gnn_out_dim, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, gnn_out_dim)

    def forward(self, x, edge_index):
        emb = self.encoder(x, edge_index)
        z = self.projection(emb, proj_act=self.config['proj_act'])
        return emb, z
    
    def projection(self, z, proj_act = 'elu'):
        z = getattr(F, proj_act)(self.fc1(z))
        return self.fc2(z)

    def loss_func(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
        if batch_size == 0:
            l1 = self.semi_loss(z1, z2)
            l2 = self.semi_loss(z2, z1)
        else:
            l1 = self.batched_semi_loss(z1, z2, batch_size)
            l2 = self.batched_semi_loss(z2, z1, batch_size)        
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()
        return ret
    

class NLGCN(nn.Module):
    def __init__(self, config):
        super(NLGCN, self).__init__()
        n_gnn_layers = config['n_gnn_layers']
        self.gnn_list = nn.ModuleList()
        out_dim = config['num_cls']
        in_dim = config['num_featss']
        for i in range(n_gnn_layers):
            if i == n_gnn_layers-1:
                hid_dim = out_dim
            self.gnn_list.append(GCNConv(in_dim, hid_dim))
            in_dim = hid_dim
        self.proj = nn.Linear(out_dim, 1)
        self.kernel = config['kernel']
        self.conv1d = nn.Conv1d(out_dim, out_dim, kernel_size= self.kernel, padding = int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(out_dim, out_dim,kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.lin = nn.Linear(2 * out_dim, out_dim)
        self.config = config

    def reset_parameters(self):
        for gnn in self.gnn_list:
            gnn.reset_parameters()
        self.proj.reset_parameters()
        self.conv1d.reset_parameters()
        self.conv1d_2.reset_parameters()
        self.lin.reset_parameters()
    
    def forward(self, data):
        x, edge_index  = data.x, data.edge_index
        
        h = F.relu(self.gnn_list[0](x, edge_index))
        for i in range(1, len(self.gnn_list)-1):
            h = F.dropout(h, p=self.config['dropout1'], training=self.training)
            h = F.relu(self.gnn_list[i](h, edge_index))
        h = F.dropout(h, p=self.config['dropout1'], training=self.training)
        h1 = self.gnn_list[-1](h, edge_index) # GNN 输出的node embeddings
        g_score = self.proj(h1)  # shape: [num_nodes, 1]   (N x D) (D x 1) -> (N x 1)  表示每个节点与 (Dx1)维向量的相似度 
        g_score_sorted, sorted_idx = torch.sort(g_score, dim=0)  # 将节点按照与(D x 1)维的Auxiliary vector 按升序排列 driving a good sorting
        _, inverse_idx = torch.sort(sorted_idx, dim=0) # 将排序后的节点按照原来的顺序排列 即原始节点的index 从1 到N
        
        # 在得到当前依据排名的node embedding h1[sorted_idx]后
        # 需要自适应调整
        sorted_x = g_score_sorted * h1[sorted_idx].squeeze()   # 按照相似度排序后的节点embedding 乘以 每个节点与Auxiliary vector的attention score
        
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0)





        

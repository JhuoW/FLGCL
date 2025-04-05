import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, GATConv

# class GCN(nn.Module):
#     def __init__(self, n_feat=10, n_dims=128, n_clss=10, args=None):
#         super().__init__()
#         self.n_feat = n_feat
#         self.n_dims = n_dims
#         self.n_clss = n_clss
#         self.args = args

#         self.conv1 = GCNConv(self.n_feat, self.n_dims, cached=False)
#         self.conv2 = GCNConv(self.n_dims, self.n_dims, cached=False)
#         self.clsif = nn.Linear(self.n_dims, self.n_clss)

#     def forward(self, data):
#         x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
#         x = F.relu(self.conv1(x, edge_index, edge_weight))
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index, edge_weight)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.clsif(x)
#         return x
    
# class Encoder(nn.Module):
#     def __init__(self, in_dim: int, hid_dim: int, out_dim: int, activation, BaseModel=GCNConv, k: int = 2) :
#         super(Encoder, self).__init__()
#         self.k = k
#         self.conv_list = nn.ModuleList()
#         # self.conv_list.append(BaseModel(in_channels, 2 * out_channels))
#         for i in range(k):  # 0, 1
#             if i == k-1:
#                 hid_dim = out_dim
#             self.conv_list.append(BaseModel(in_dim, hid_dim))
#             in_dim = hid_dim
#         self.activation = activation

#     def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
#         for i in range(self.k):
#             x = self.activation(self.conv_list[i](x, edge_index))
#         return x

# class Augmentator(nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)

# class GRACE(nn.Module):
#     def __init__(self, config):
#         super(GRACE, self).__init__()
#         self.config = config
#         gnn_hid_dim = config['gnn_hid_dim']
#         gnn_out_dim = config['gnn_out_dim']
#         gnn_act = ({'relu': F.relu, 'prelu': nn.PReLU(), 'elu': F.elu})[config['gnn_act']]
#         BaseGNN = getattr(pyg_nn, config.get('BaseGNN', 'GCNConv'))
#         n_gnn_layers = config['n_gnn_layers']
#         num_proj_hidden = config['num_proj_hidden']
#         in_dim  = config['num_feats']
#         self.encoder = Encoder(in_dim, gnn_hid_dim, gnn_out_dim, gnn_act, BaseGNN, k = n_gnn_layers)

#         self.fc1 = torch.nn.Linear(gnn_out_dim, num_proj_hidden)
#         self.fc2 = torch.nn.Linear(num_proj_hidden, gnn_out_dim)

#     def forward(self, x, edge_index):
#         emb = self.encoder(x, edge_index)
#         z = self.projection(emb, proj_act=self.config['proj_act'])
#         return emb, z
    
#     def projection(self, z, proj_act = 'elu'):
#         z = getattr(F, proj_act)(self.fc1(z))
#         return self.fc2(z)

#     def loss_func(self, z1: torch.Tensor, z2: torch.Tensor, mean: bool = True, batch_size: int = 0):
#         if batch_size == 0:
#             l1 = self.semi_loss(z1, z2)
#             l2 = self.semi_loss(z2, z1)
#         else:
#             l1 = self.batched_semi_loss(z1, z2, batch_size)
#             l2 = self.batched_semi_loss(z2, z1, batch_size)        
#         ret = (l1 + l2) * 0.5
#         ret = ret.mean() if mean else ret.sum()
#         return ret


class NLGNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        n_gnn_layers = config['n_gnn_layers']
        self.gnn_list = nn.ModuleList()
        out_dim = config['num_cls']
        in_dim = config['num_feats']
        hid_dim = config['gnn_hid_dim']
        for i in range(n_gnn_layers):
            if i == n_gnn_layers-1:
                hid_dim = out_dim

            if config.get('BaseGNN', 'GCNConv') == 'GCNConv':
                self.gnn_list.append(GCNConv(in_dim, hid_dim))
            elif config.get('BaseGNN', 'GCNConv') == 'GATConv':
                if i == 0:
                    self.gnn_list.append(GATConv(in_dim, hid_dim, heads = config['hidden_heads'], dropout=config['dropout1']))
                else:
                    self.gnn_list.append(GATConv(hid_dim * config['hidden_heads'], out_dim, heads = 1, dropout=config['dropout1'], concat = False))
            in_dim = hid_dim
        self.proj = nn.Linear(out_dim, 1)
        self.kernel = config['kernel']
        self.conv1d = nn.Conv1d(out_dim, out_dim, kernel_size= self.kernel, padding = int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(out_dim, out_dim, kernel_size=self.kernel, padding = int((self.kernel-1)/2))
        self.lin = nn.Linear(2 * out_dim, out_dim)
        

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
        # GNN 输出的node embeddings
        h1 = self.gnn_list[-1](h, edge_index) 
        # shape: [num_nodes, 1]   (N x D) (D x 1) -> (N x 1)  表示每个节点与 (Dx1)维 Auxiliary 向量的相似度 
        g_score = self.proj(h1)  
        # learn the optimal sorting
        # sorted_idx: 所有节点和auxiliary vector的相似度按升序排列id  例如 [2,3,1] 表示节点2与auxiliary vector的相似度最低， 节点1的相似度最高
        g_score_sorted, sorted_idx = torch.sort(g_score, dim=0)  # 将节点按照与(D x 1)维的Auxiliary vector 按升序排列 driving a good sorting
        # sorted_idx = [2 3 1] -> [1 2 3]   inverse_idx: [3 1 2]  表示按照原始的节点idx, 每个节点应该处于Auxiliary vector的第几位 
        _, inverse_idx = torch.sort(sorted_idx, dim=0) 
        
        # 在得到当前依据排名的node embedding: h1[sorted_idx]后
        # 一个感受野中的节点tend to have similar attention scores, 所以对attention scores做聚合
        sorted_x = g_score_sorted * h1[sorted_idx].squeeze()   # 按照相似度排序后的节点embedding 乘以 每个节点与Auxiliary vector的attention score
        
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0)  # [1, D, N]  为了做conv1d  D个通道，每个通道N维，每个通道上都要做1d卷积， 再pooling

        # 1d convolution
        sorted_x = F.relu(self.conv1d(sorted_x)) 
        sorted_x = F.dropout(sorted_x, p=self.config['dropout2'], training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)  # [num_nodes, num_classes]
        h2       = sorted_x[inverse_idx].squeeze()  # 恢复到原始的node idx
        # print(h1.shape, h2.shape) # torch.Size([248, 7]) torch.Size([248, 1, 7])
        out = torch.cat([h1, h2], dim=1)
        out = self.lin(out)
        return out



        


class NLGNN2(nn.Module):
    def __init__(self, n_gnn_layers, in_dim, hid_dim, out_dim, kernel, dropout1, dropout2):
        super(NLGNN2, self).__init__()
        
        self.gnn_list = nn.ModuleList()
        self.dropout1 = dropout1
        self.dropout2 = dropout2
        # out_dim = config['num_cls']
        # in_dim = config['num_feats']
        # hid_dim = config['gnn_hid_dim']
        # n_gnn_layers = config['n_gnn_layers']
        for i in range(n_gnn_layers):
            if i == n_gnn_layers-1:
                hid_dim = out_dim

            self.gnn_list.append(GCNConv(in_dim, hid_dim))
            in_dim = hid_dim
        self.proj = nn.Linear(out_dim, 1)
        self.kernel = kernel
        self.conv1d = nn.Conv1d(out_dim, out_dim, kernel_size= self.kernel, padding = int((self.kernel-1)/2))
        self.conv1d_2 = nn.Conv1d(out_dim, out_dim,kernel_size=self.kernel, padding=int((self.kernel-1)/2))
        self.lin = nn.Linear(2 * out_dim, out_dim)
        

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
            h = F.dropout(h, p=self.dropout1, training=self.training)
            h = F.relu(self.gnn_list[i](h, edge_index))
        h = F.dropout(h, p=self.dropout1, training=self.training)
        h1 = self.gnn_list[-1](h, edge_index) # GNN 输出的node embeddings
        g_score = self.proj(h1)  # shape: [num_nodes, 1]   (N x D) (D x 1) -> (N x 1)  表示每个节点与 (Dx1)维向量的相似度 
        # learn the optimal sorting
        g_score_sorted, sorted_idx = torch.sort(g_score, dim=0)  # 将节点按照与(D x 1)维的Auxiliary vector 按升序排列 driving a good sorting
        # sorted_idx = [2 3 1] -> [1 2 3]   inverse_idx: [3 1 2]   
        _, inverse_idx = torch.sort(sorted_idx, dim=0) # 将排序后的节点按照原来的顺序排列 即原始节点的index 从1 到N  
        
        # 在得到当前依据排名的node embedding h1[sorted_idx]后
        # 一个感受野中的节点tend to have similar attention scores, 所以对attention scores做聚合
        sorted_x = g_score_sorted * h1[sorted_idx].squeeze()   # 按照相似度排序后的节点embedding 乘以 每个节点与Auxiliary vector的attention score
        
        sorted_x = torch.transpose(sorted_x, 0, 1).unsqueeze(0)  # [1, D, N]  为了做conv1d  D个通道，每个通道N维，每个通道上都要做1d卷积， 再pooling

        # 1d convolution
        sorted_x = F.relu(self.conv1d(sorted_x)) 
        sorted_x = F.dropout(sorted_x, p=self.dropout2, training=self.training)
        sorted_x = self.conv1d_2(sorted_x)
        sorted_x = torch.transpose(sorted_x.squeeze(), 0, 1)  # [num_nodes, num_classes]
        h2       = sorted_x[inverse_idx]  # 恢复到原始的node idx

        out = torch.cat([h1, h2], dim=1)
        out = self.lin(out)
        return out



        
class GCN(torch.nn.Module):
    def __init__(self, n_gnn_layers, in_dim, hid_dim, out_dim, kernel, dropout1, dropout2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hid_dim, normalize= False)
        self.conv2 = GCNConv(hid_dim, out_dim,
                             normalize=False)
        self.dropout1 = dropout1
        

    def forward(self, x, edge_index, edge_weight=None):
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = F.dropout(x, p=self.dropout1, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
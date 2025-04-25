import torch
import torch.nn as nn
import torch.nn.functional as F
from importlib import import_module
import torch_geometric.nn as pyg_nn
from torch_geometric.nn import GCNConv, GATConv



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
    def __init__(self, config):
        super().__init__()
        self.conv1 = GCNConv(config['num_feats'], config['gnn_hid_dim'])
        self.conv2 = GCNConv(config['gnn_hid_dim'], config['gnn_hid_dim'])
        self.dropout1 = config['dropout1']
        self.projection = nn.Linear(config['gnn_hid_dim'], config['num_cls'])
        

    def forward(self, data):
        # x = F.dropout(x, p=self.dropout1, training=self.training)
        x, edge_index  = data.x, data.edge_index
        h = self.conv1(x, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout1, training=self.training)
        h = self.conv2(h, edge_index)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout1, training=self.training)
        h = self.projection(h)
        return h


class LocalModel(nn.Module):
    def __init__(self, config, anchor_dim: int, sigma: float):
        super().__init__()
        self.conv1 = GCNConv(config['num_feats'], config['gnn_hid_dim'])
        self.conv2 = GCNConv(config['gnn_hid_dim'], config['gnn_hid_dim'])
        self.aux   = nn.Parameter(torch.randn(anchor_dim))  #  \boldsymbol a
        self.clf   = nn.Linear(2 * config['gnn_hid_dim'], config['num_cls'])
        self.sigma = sigma

    # ---------- kernel‑aggregator ------------------------------------
    def _kernel_aggregate(self, h: torch.Tensor) -> torch.Tensor:
        """
        h :  [N, d]  node embeddings
        returns z :  [N, d]  kernel‑smoothed embeddings
        """
        a = F.normalize(self.aux, dim=0)                          # unit vector
        score = F.cosine_similarity(h, a.unsqueeze(0), dim=-1)    # s_i  (N)
        diff  = score.unsqueeze(0) - score.unsqueeze(1)           # (N,N)
        kappa = torch.exp(-(diff ** 2) / (self.sigma ** 2))       # eq.(3)
        z     = (kappa @ h) / kappa.sum(dim=1, keepdim=True)
        return z

    # ---------- forward pass -----------------------------------------
    def forward(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=self.dropout1, training=self.training)
        h = self.conv2(h, edge_index)                              # h_i
        z = self._kernel_aggregate(h)                             # z_i
        out = self.clf(torch.cat([h, z], dim=-1))                 # π_i
        return out
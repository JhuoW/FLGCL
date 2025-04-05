import torch
import random
import numpy as np
import os
import metispy as metis
import os.path as osp
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj, dense_to_sparse, to_networkx
from torch_geometric.transforms import BaseTransform
from torch_geometric.datasets import Planetoid, Amazon
from torch_geometric.utils import to_scipy_sparse_matrix
import torch_geometric.transforms as T
from ogb.nodeproppred import PygNodePropPredDataset



def torch_save(base_dir, filename, data):
    # base_dir: dataset/Cora/disjoint/n_clients 
    # filename: train.pt
    # data = {'data': data}
    os.makedirs(base_dir, exist_ok = True)
    fpath = osp.join(base_dir, filename)    
    torch.save(data, fpath)


def split_train(args, data, ratio_train, n_clients):
    n_data = data.num_nodes
    ratio_test = (1-ratio_train) / 2 # ratio_ratio = 20%, val/test = 40%
    n_train = round(n_data * ratio_train)
    n_test = round(n_data * ratio_test)
    permuted_indices = torch.randperm(n_data)
    train_indices = permuted_indices[:n_train]
    test_indices = permuted_indices[n_train:n_train+n_test]
    val_indices = permuted_indices[n_train+n_test:]
    data.train_mask.fill_(False)
    data.test_mask.fill_(False)
    data.val_mask.fill_(False)

    data.train_mask[train_indices] = True
    data.test_mask[test_indices] = True
    data.val_mask[val_indices] = True
    # dataset/Cora/disjoint/n_clients/train.pt
    torch_save(osp.join(args.data_path, args.dataset, f'{args.mode}/{n_clients}'), 'train.pt', {'data': data})
    torch_save(osp.join(args.data_path, args.dataset, f'{args.mode}/{n_clients}'), 'test.pt', {'data': data})
    torch_save(osp.join(args.data_path, args.dataset, f'{args.mode}/{n_clients}'), 'val.pt', {'data': data})
    print(f'splition done, n_train: {n_train}, n_test: {n_test}, n_val: {len(val_indices)}')
    return data



class LargestConnectedComponents(BaseTransform):
    r"""Selects the subgraph that corresponds to the
    largest connected components in the graph.

    Args:
        num_components (int, optional): Number of largest components to keep
            (default: :obj:`1`)
    """
    def __init__(self, num_components: int = 1):
        self.num_components = num_components

    def __call__(self, data: Data) -> Data:
        import numpy as np
        import scipy.sparse as sp

        adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes)

        num_components, component = sp.csgraph.connected_components(adj)

        if num_components <= self.num_components:
            return data

        _, count = np.unique(component, return_counts=True)
        subset = np.in1d(component, count.argsort()[-self.num_components:])

        return data.subgraph(torch.from_numpy(subset).to(torch.bool))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num_components})'

def get_data(dataset, data_path):
    if dataset in ['Cora', 'CiteSeer', 'PubMed']:
        dataset = Planetoid(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))
        data = dataset[0]
    elif dataset in ['Computers', 'Photo']:
        dataset = Amazon(data_path, dataset, transform=T.Compose([LargestConnectedComponents(), T.NormalizeFeatures()]))
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
    elif dataset in ['ogbn-arxiv']:
        dataset = PygNodePropPredDataset(dataset, root=data_path, transform=T.Compose([T.ToUndirected(), LargestConnectedComponents()]))
        data = dataset[0]
        data.train_mask, data.val_mask, data.test_mask \
            = torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool), torch.zeros(data.num_nodes, dtype=torch.bool)
        data.y = data.y.view(-1)
    return data, dataset

class DataBuild:
    def __init__(self, 
                 args, 
                 config,
                 ratio_train = 0.2,  # 20% as training nodes
                 seed = 1234):

        self.args = args
        self.ratio_train = ratio_train
        self.config = config
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        # self.data = self.load_data()
        data, dataset = get_data(self.args.dataset, self.args.data_path)
        self.oridata = data
        self.oridataset = dataset
        self.num_feats = data.num_features
        self.num_classes = dataset.num_classes

    
    def load_data(self, n_clients = 0, n_comms = 0, n_clien_per_comm = 0):   # n_clients = [5, 10, 20]
        splited_data = split_train(self.args, self.oridata, self.ratio_train, n_clients = n_clients if self.args.mode == 'disjoint' else n_comms * n_clien_per_comm)
        self.data = splited_data

    def split_subgraphs(self, n_clients = 10, n_comms = 5, n_clien_per_comm = 2):  # n_clients = 5, 10, 20
        nx_G = to_networkx(self.data)
        # membership indicate每个节点的cluster idx
        n_cuts, membership = metis.part_graph(nx_G, n_clients if self.args.mode == 'disjoint' else n_comms)  # partiton graph to n_clients subgraph, with minimal number of cross-partition edges
        assert len(list(set(membership))) == n_clients if self.args.mode == 'disjoint' else n_comms 
        print(f'graph partition done, metis, n_partitions: {len(list(set(membership)))}, n_lost_edges: {n_cuts}')        
        adj = to_dense_adj(self.data.edge_index)[0]
        if self.args.mode == 'disjoint':
            for client_id in range(n_clients):
                # 当前client中的节点 idx 和 节点个数
                client_node_idx = np.where(np.array(membership) == client_id)[0]
                client_node_idx = list(client_node_idx)
                client_num_nodes = len(client_node_idx)

                client_edge_index = []
                client_adj = adj[client_node_idx][:, client_node_idx] # dense adj of client
                client_edge_index, _ = dense_to_sparse(client_adj)
                client_edge_index = client_edge_index.T.tolist()
                client_num_edges = len(client_edge_index)
                
                client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
                client_x = self.data.x[client_node_idx]
                client_y = self.data.y[client_node_idx]
                client_train_mask = self.data.train_mask[client_node_idx]
                client_val_mask = self.data.val_mask[client_node_idx]
                client_test_mask = self.data.test_mask[client_node_idx]
                client_subgraph = Data(x = client_x,
                                    y = client_y,
                                    edge_index=client_edge_index.t().contiguous(),
                                    train_mask = client_train_mask,
                                    val_mask = client_val_mask,
                                    test_mask = client_test_mask)
                assert torch.sum(client_train_mask).item() > 0
                # 保存每个client
                torch_save(osp.join(self.args.data_path, self.args.dataset, f'{self.args.mode}/{n_clients}'), f'partition_{client_id}.pt', {
                        'client_data': client_subgraph,
                        'client_id': client_id})
                print(f'client_id: {client_id}, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')
        else:
            for comm_id in range(n_comms):
                # 每个community 有5个clients
                for client_id in range(n_clien_per_comm):   # 每个community 中的所有clients 5个 每个client占用community一半的nodes
                    client_node_idx = np.where(np.array(membership) == comm_id)[0]  # 属于每个community中的所有节点id
                    client_node_idx = list(client_node_idx)
                    client_num_nodes = len(client_node_idx)
                    
                    client_node_idx = random.sample(client_node_idx, client_num_nodes // 2)  # 选出community中一半的节点
                    client_num_nodes = len(client_node_idx)

                    client_edge_index = []
                    client_adj = adj[client_node_idx][:, client_node_idx]
                    client_edge_index, _ = dense_to_sparse(client_adj)
                    client_edge_index = client_edge_index.T.tolist()
                    client_num_edges = len(client_edge_index)

                    client_edge_index = torch.tensor(client_edge_index, dtype=torch.long)
                    client_x = self.data.x[client_node_idx]
                    client_y = self.data.y[client_node_idx]
                    client_train_mask = self.data.train_mask[client_node_idx]
                    client_val_mask = self.data.val_mask[client_node_idx]
                    client_test_mask = self.data.test_mask[client_node_idx]
                    client_subgraph = Data(
                        x = client_x,
                        y = client_y,
                        edge_index = client_edge_index.t().contiguous(),
                        train_mask = client_train_mask,
                        val_mask = client_val_mask,
                        test_mask = client_test_mask
                    )
                    assert torch.sum(client_train_mask).item() > 0
                    # 每个community有5个互相有重叠部分的clients 母目录为从client数量 比如有2个community，那么就有10个client 然后client文件是逐个编号的，前5个是在一个comm内，后5个在一个comm内
                    torch_save(osp.join(self.args.data_path, self.args.dataset, f'{self.args.mode}/{n_comms * n_clien_per_comm}'), f'partition_{comm_id*n_clien_per_comm+client_id}.pt', {
                        'client_data': client_subgraph,
                        'client_id': client_id})
                    print(f'client_id: {comm_id*n_clien_per_comm+client_id}, n_train_node: {client_num_nodes}, n_train_edge: {client_num_edges}')

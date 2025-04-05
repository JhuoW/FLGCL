import os.path as osp
import torch

class DataLoader:
    def __init__(self, args):
        self.args = args
        self.n_workers  = 1
        self.client_id = None

        from torch_geometric.loader import DataLoader
        self.DataLoader = DataLoader
    
    def switch(self, client_id): # 要切换到的client
        if not self.client_id == client_id:  # 切换到下一个client
            self.client_id = client_id
            self.partition = get_data(self.args, client_id=client_id)  

            # 一个partition是一个client，一个batch是一个client的数据
            # 一个partition_loader一次迭代得到一个client的数据和id
            self.patition_loader = self.DataLoader(dataset=self.partition,  batch_size=1, 
                shuffle=False, num_workers=self.n_workers, pin_memory=False)


def load_data(base_dir, filename):
    fpath = osp.join(base_dir, filename)
    return torch.load(fpath, map_location=torch.device('cpu'), weights_only=False)

def get_data(args, client_id):
    return [
        load_data(
            osp.join(args.data_path, args.dataset, f'{args.mode}', f'{args.n_clients}'),  # datasets
            f'partition_{client_id}.pt'
        )['client_data']
    ]

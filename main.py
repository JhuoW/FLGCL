from parser import Parser
import torch
from misc.logger import Logger
import sys
import os.path as osp
from misc.random_seeder import set_random_seed
import yaml
from misc.utils import *
import pathlib
from datahelper.generators import DataBuild
from datetime import datetime
from multiprocs import ParentProcess

def main(args, config):
    if args.model == 'FedAvg':
        # from models.fedpubclient import Client
        from models.client import Client
        from models.server import Server
    elif args.model == 'FedAux':
        from models.server import Server
        from models.client import Client
    else:
        raise ValueError('No Model')
    pp = ParentProcess(args, config, Server, Client)
    pp.start() # 开启主进程


if __name__ == '__main__':
    args = Parser().parse()
    # torch.cuda.set_device(args.gpu_id)
    logger = Logger(args=args, mode = [print])  
    logger.add_line = lambda : logger.log("-" * 50)
    logger.log(" ".join(sys.argv))
    logger.add_line()
    logger.log()
    
    config_path = osp.join('config/', args.dataset + '.yml')
    def get_config(config_path="config.yml"):
        with open(config_path, "r") as setting:
            config = yaml.load(setting, Loader=yaml.FullLoader)
        return config

    config = get_config(config_path)
    dataset_config = config['dataset']
    config = config[args.model]
    config['dname'] = args.dataset

    print_config(config)
    pathlib.Path(args.data_path).mkdir(parents=True, exist_ok=True)
    preprocess_path = osp.join(args.data_path, args.dataset, args.mode) # datasets/Cora/disjoint
    databuild = DataBuild(args, config, dataset_config['ratio_train'], dataset_config['seed'])
    if not osp.exists(preprocess_path):
        print('Generating Clients...')
        if args.mode == 'disjoint':
            for n_clients in dataset_config['clients']:  # [5, 10, 20]
                databuild.load_data(n_clients)
                databuild.split_subgraphs(n_clients=n_clients)
        elif args.mode == 'overlapping':
            for n_comms in dataset_config['comms']:
                databuild.split_subgraphs(n_comms=n_comms, n_clien_per_comm=dataset_config['n_clien_per_comm'])
        else:
            raise ValueError('No Mode (Disjoint/Overlapping)')
    else:
        print('Clients Exists')
        

    config['num_feats'] = databuild.num_feats
    config['num_cls']  = databuild.num_classes

    if config.get('seed',-1)     > 0:
        set_random_seed(config['seed'])
        logger.log ("Seed set. %d" % (config['seed']))
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    trial = osp.join(f'{args.dataset}', f'{args.mode}', f'{args.n_clients}', f'{now}_{args.model}_{config["Backbone"]}')
    args.checkpt_path = f'checkpoints/{trial}'
    args.log_path = f'logs/{trial}'

    main(args, config)
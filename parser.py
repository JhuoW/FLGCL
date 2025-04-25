import argparse


class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()

    def set_arguments(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0,1,2')
        self.parser.add_argument('--seed', type=int, default=42)
        self.parser.add_argument('--dataset', type = str, default='Cora')
        self.parser.add_argument('--n_workers', type = int, default=10)
        self.parser.add_argument('--model',  type = str,  default ='FedAvg', choices=['FedAvg', 'FedAux'])
        self.parser.add_argument('--data_path', type = str, default='datasets')
        self.parser.add_argument('--mode', type=str, default='disjoint', choices=['disjoint', 'overlapping'])
        self.parser.add_argument('--n_clients', type=int, default=10)
        self.parser.add_argument('--log_path', type=str, default='log/')
        self.parser.add_argument('--frac', type=float, default=1.0)
        # self.parser.add_argument('--checkpt_path', type=str, default='checkpoint/')

    
    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
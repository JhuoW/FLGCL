from datahelper.generators import DataBuild
from argparse import ArgumentParser



def generate_clients(args):
    if args.mode == 'disjoint':
        disjoint_data = DataBuild(args, )



if __name__ == '__main__':
    parser = ArgumentParser()

    
    parser.add_argument('--save_model',type = bool, default = False)

    args = parser.parse_args()
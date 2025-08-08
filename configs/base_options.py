import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class BaseOptions():
    def __init__(self):
        pass

    def initialize(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # base configs
        parser.add_argument('--exp_name',   type=str, default='')
        parser.add_argument('--gpu_or_cpu',   type=str, default='gpu')
        parser.add_argument('--data_path',    type=str, default='./ho3d/')
        parser.add_argument('--dataset',      type=str, default='ov3d_dataset')
        parser.add_argument('--batch_size',   type=int, default=16)
        parser.add_argument('--workers',      type=int, default=8)
        
        parser.add_argument('--use_checkpoint',   type=str2bool, default='False')
        parser.add_argument('--num_deconv',     type=int, default=3)
        parser.add_argument('--num_filters', nargs='+', type=int, default=[32, 32, 32])
        parser.add_argument('--deconv_kernels', nargs='+', type=int, default=[2, 2, 2])

        parser.add_argument('--dino', action='store_true') 
        
        return parser

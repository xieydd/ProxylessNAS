'''
@Description: Config class for config
@Author: xieydd
@Date: 2019-08-13 14:01:03
@LastEditTime: 2019-10-18 17:58:28
@LastEditors: Please set LastEditors
'''
import argparse
import os
import torch
from functools import partial


def get_parser(name):
    """ make default formatted parser """
    parser = argparse.ArgumentParser(
        name, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # print default value always
    parser.add_argument = partial(parser.add_argument, help=' ')
    return parser


'''
@description: for mutil gpus train in pytorch
@param string "0,1,2,3" 
@return: gpu list [0, 1, 2, 3]
'''


def parse_gpus(gpus):
    if gpus == 'all':
        return list(range(torch.cuda.device_count()))
    else:
        return [int(s) for s in gpus.split(',')]


'''
@description: Base config class include a print params function and as_markdown function
@param {type} 
@return: 
'''


class BaseConfig(argparse.Namespace):
    def print_params(self, prtf=print):
        prtf("")
        prtf("Parameters:")
        for attr, value in sorted(vars(self).items()):
            prtf("{}={}".format(attr.upper(), value))
        prtf("")

    def as_markdown(self):
        """ Return configs as markdown format """
        text = "|name|value|  \n|-|-|  \n"
        for attr, value in sorted(vars(self).items()):
            text += "|{}|{}|  \n".format(attr, value)
        return text


'''
@description: For NAS model,may need to config class for model: SearchConfig for Search Network model \
     and AugmentConfig stack searched out model to a big model and train
@param {type} 
@return: 
'''


class SearchConfig(BaseConfig):
    def build_parser(self):
        parser = get_parser("Search config")
        parser.add_argument('--name', required=True)
        parser.add_argument('--dataset', required=True,
                            help='CIFAR10 / MNIST / FashionMNIST / ImageNet')
        parser.add_argument('--batch_size', type=int,
                            default=64, help='batch size')
        parser.add_argument('--w_lr', type=float,
                            default=0.025, help='lr for weights')
        parser.add_argument('--w_lr_min', type=float,
                            default=0.001, help='minimum lr for weights')
        parser.add_argument('--w_momentum', type=float,
                            default=0.9, help='momentum for weights')
        parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                            help='weight decay for weights')
        parser.add_argument('--w_grad_clip', type=float, default=5.,
                            help='gradient clipping for weights')
        parser.add_argument('--print_freq', type=int,
                            default=50, help='print frequency')
        parser.add_argument('--gpus', default='0', help='gpu device ids separated by comma. '
                            '`all` indicates use all gpus.')
        parser.add_argument('--epochs', type=int, default=50,
                            help='# of training epochs')
        parser.add_argument('--init_channels', type=int, default=16)
        parser.add_argument('--layers', type=int,
                            default=8, help='# of layers')
        parser.add_argument('--seed', type=int, default=2, help='random seed')
        parser.add_argument('--cutout', action='store_true',
                            default=False, help='use cutout')
        parser.add_argument('--cutout_length', type=int,
                            default=16, help='cutout length')
        parser.add_argument('--drop_path_prob', type=float,
                            default=0.3, help='drop path probability')
        parser.add_argument('--grad_clip', type=float,
                            default=5, help='gradient clipping')
        parser.add_argument('--train_portion', type=float,
                            default=0.5, help='portion of training data')
        parser.add_argument('--val_portion', type=float,
                            default=0.5, help='portion of validate data')
        parser.add_argument("--local_rank", default=0, type=int)
        parser.add_argument('--unrolled', action='store_true',
                            default=False, help='use one-step unrolled validation loss')
        parser.add_argument('--workers', type=int,
                            default=4, help='# of workers')
        parser.add_argument('--label_smooth', type=float,
                            default=0.1, help='label smoothing')
        parser.add_argument('--width_mult', type=float, default=1.0)
        parser.add_argument('--resume', action='store_true')
        # architecture search config
        """ arch search algo and warmup """
        parser.add_argument('--arch_algo', type=str,
                            default='grad', choices=['grad', 'rl'])
        parser.add_argument('--warmup_epochs', type=int, default=40)
        parser.add_argument('--start_epoch', type=int, default=0)
        """ shared hyper-parameters """
        parser.add_argument('--arch_init_type', type=str,
                            default='normal', choices=['normal', 'uniform'])
        parser.add_argument('--arch_init_ratio', type=float, default=1e-3)
        parser.add_argument('--arch_opt_type', type=str,
                            default='adam', choices=['adam'])
        parser.add_argument('--alpha_lr', type=float,
                            default=1e-3, help='lr for alpha')
        parser.add_argument('--alpha_weight_decay', type=float, default=0,
                            help='weight decay for alpha')
        parser.add_argument('--arch_adam_beta1', type=float,
                            default=0)  # arch_opt_param
        parser.add_argument('--arch_adam_beta2', type=float,
                            default=0.999)  # arch_opt_param
        parser.add_argument('--arch_adam_eps', type=float,
                            default=1e-8)  # arch_opt_param
        parser.add_argument('--target_hardware', type=str, default=None,
                            choices=['mobile', 'cpu', 'gpu8', 'flops', None])
        """ Grad hyper-parameters """
        parser.add_argument(
            '--grad_update_arch_param_every', type=int, default=5)
        parser.add_argument('--grad_update_steps', type=int, default=1)
        parser.add_argument('--grad_binary_mode', type=str,
                            default='full_v2', choices=['full_v2', 'full', 'two'])
        parser.add_argument('--grad_data_batch', type=int, default=None)
        parser.add_argument('--grad_reg_loss_type', type=str,
                            default='mul#log', choices=['add#linear', 'mul#log'])
        parser.add_argument('--grad_reg_loss_lambda',
                            type=float, default=1e-1)  # grad_reg_loss_params
        parser.add_argument('--grad_reg_loss_alpha', type=float,
                            default=0.2)  # grad_reg_loss_params
        parser.add_argument('--grad_reg_loss_beta', type=float,
                            default=0.3)  # grad_reg_loss_params

        """ RL hyper-parameters """
        parser.add_argument('--rl_batch_size', type=int, default=10)
        parser.add_argument('--rl_update_per_epoch', action='store_true')
        parser.add_argument('--rl_update_steps_per_epoch',
                            type=int, default=300)
        parser.add_argument('--rl_baseline_decay_weight',
                            type=float, default=0.99)
        parser.add_argument('--rl_tradeoff_ratio', type=float, default=0.1)
        return parser

    def __init__(self):
        parser = self.build_parser()
        args = parser.parse_args()
        super().__init__(**vars(args))

        self.data_path = './data/'
        self.path = os.path.join('searchs', self.name)
        self.plot_path = os.path.join(self.path, 'plots')
        self.gpus = parse_gpus(self.gpus)

import torch
import argparse

from gaussian_core.provider import EndoDataset
from gaussian_core.utils import *
from gaussian_core.gaussian_model import GaussianModel

try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except AttributeError as e:
    print('Info. This pytorch version is not support with tf32.')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--data_range', type=int, nargs='*', default=[0, -1], help="data range to use")
    parser.add_argument('--workspace', type=str, default='workspace')
    parser.add_argument('--seed', type=int, default=0)

    ### training options
    parser.add_argument('--coarse_iters', type=int, default=3000, help="training iters")
    parser.add_argument('--fine_iters', type=int, default=60000, help="training iters")

    parser.add_argument('--percent_dense', type=float, default=0.01, help="percent of dense points")
    parser.add_argument('--position_lr_init', type=float, default=0.00016, help="initial learning rate")
    parser.add_argument('--position_lr_final', type=float, default=0.0000016, help="final learning rate")
    parser.add_argument('--position_lr_delay_mult', type=float, default=0.01, help="delay multiplier for learning rate")
    parser.add_argument('--position_lr_max_steps', type=int, default=1000000, help="max steps for learning rate schedule")
    parser.add_argument('--grid_lr_init', type=float, default=0.00015, help="initial learning rate")
    parser.add_argument('--grid_lr_final', type=float, default=0.000015, help="final learning rate")
    parser.add_argument('--deformation_lr_init', type=float, default=0.000015, help="initial learning rate")
    parser.add_argument('--deformation_lr_final', type=float, default=0.0000015, help="final learning rate")
    parser.add_argument('--deformation_lr_delay_mult', type=float, default=0.01, help="delay multiplier for learning rate")
    parser.add_argument('--deformation_lr_max_steps', type=int, default=1000000, help="max steps for learning rate schedule")
    parser.add_argument('--feature_lr', type=float, default=0.0025, help="initial learning rate")
    parser.add_argument('--opacity_lr', type=float, default=0.05, help="initial learning rate")
    parser.add_argument('--scaling_lr', type=float, default=0.005, help="initial learning rate")
    parser.add_argument('--rotation_lr', type=float, default=0.001, help="initial learning rate")

    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)

    gaussians = GaussianModel(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = EndoDataset(opt, device=device, type='train').dataloader()

    if opt.workspace is not None:
        os.makedirs(opt.workspace, exist_ok=True)

    training(opt, dataloader, gaussians)
    print("\nTraining complete.")
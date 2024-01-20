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
    parser.add_argument('--model_path', type=str, default='')

    opt = parser.parse_args()
    print(opt)
    
    seed_everything(opt.seed)

    gaussians = GaussianModel(opt)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = EndoDataset(opt, device=device, type='test').dataloader()

    testing(opt, dataloader, gaussians)
    print("\nInference complete.")
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np

from losses import NCCLoss, grad_loss3d
from net.solver import Solver
# from net.solver_hos import Solver
import warnings

from net.MISP_net import MISPnet 

# import os

# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32"
warnings.filterwarnings("ignore")


def loss_fn(pred_warp, data_fix, flow):
    loss_smi = NCCLoss(window=(9,9,9))(pred_warp, data_fix)
    loss_smooth = 0.05 * grad_loss3d(flow)
    return loss_smi, loss_smooth

def train(args):    

    # make output folder
    os.makedirs('./ournet/output', exist_ok=True)
    output_dir = './ournet/output/dirlab_%s_%s' % (args.net, args.tag)
    os.makedirs(os.path.join(output_dir), exist_ok=True)

    # build network
    net = MISPnet(args.inshape)

    print('# net parameters:', sum(param.numel() for param in net.parameters()))
    print(net)
    
    # build dataset
    # build solver
    criterion = loss_fn
    solver = Solver(net, args, args.batch_size, criterion, output_dir)
    # start training
    for i_epoch in range(args.num_epoch):
        # train
        solver.train_one_epoch()# solver.num_epoch+1
        if i_epoch % args.save_interval == 0:
            save_path = solver.save_model()
            print('save model at %s' % save_path)
            solver.test_result()
    print('end')
    # print('save')
    # solver = Solver(net, args, args.batch_size, criterion, output_dir)
    # solver.test_result()

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train bone segmentation model')
    parser.add_argument("--local_rank", default=2, type=int)# GPU
    parser.add_argument("--net", help="netname for network factory",default="dirlab_600" ,type=str)
    parser.add_argument("--num_epoch", help="number of total epoch", default=600, type=int)
    parser.add_argument("--batch_size", help="batch size", default=1, type=int)
    parser.add_argument("--train_data_root", help="training data folder", default='./dirlab',type=str)
    parser.add_argument("--save_interval", help="save interval", default=5, type=str)
    parser.add_argument("--tag", help="output name", default='default', type=str)
    parser.add_argument("--inshape", help="saving data folder", default=[128,128,128],type=list) 

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # parse augmentations
    args = parse_args()

    rank = args.local_rank
    torch.cuda.set_device(rank)

    # start
    print(args.net)
    print(args.local_rank)
    print(args.inshape[0])
    train(args)
import sys
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.utils.data import sampler
import numpy as np

from tricks import WarmUpLR
# from tensorboardX import SummaryWriter

import SimpleITK as sitk

from losses import NCCLoss, grad_loss3d,calculate_mse,dice_coefficient,hausdorff_distance,pnsr

from dataset.dirlab_dataset import DIRLabDataset
from dataset.pair_dataset import PairDataset

class CollateFn:
    # CollateFn for unsupervised training (without label)
    def __init__(self):
        pass

    def __call__(self, batch_data):
        left_list = []
        right_list = []
        for left, right in batch_data:
            left_list.append(left)
            right_list.append(right)

        data_left = torch.stack(left_list, dim=0)
        data_right = torch.stack(right_list, dim=0)

        data_left = data_left.unsqueeze(1)
        data_right = data_right.unsqueeze(1)

        return data_left, data_right

class Solver(object):

    def __init__(self, net, args, batch_size, criterion, output_dir):
        self.net = net
        train_data = DIRLabDataset(args.train_data_root,mode='train',inshape=args.inshape[0])
        train_data = PairDataset(train_data)
        val_data = DIRLabDataset(args.train_data_root,mode='val',inshape=args.inshape[0])
        val_data = PairDataset(val_data)
        print(len(train_data))
        print(len(val_data))
        self.dataset_train = train_data
        self.dataset_val = val_data
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.criterion = criterion

        self.net.cuda()
        self.batch_data = DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
            num_workers=1, collate_fn=CollateFn(), pin_memory=True, drop_last=True)
        self.val_data = DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False,
            num_workers=1, collate_fn=CollateFn(), pin_memory=True, drop_last=True)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4, weight_decay=1e-5)
        self.warm_epoch = 5
        self.max_iter = len(self.batch_data)
        self.warmup_scheduler = WarmUpLR(self.optimizer, self.max_iter * self.warm_epoch)


        # training stage
        self.num_iter = 0
        self.num_epoch = 0


    def train_one_epoch(self, iter_size=1):
        self.net.train()
        max_iter = len(self.batch_data)

        for i_batch, (data_warp, data_fix) in enumerate(self.batch_data):
            self.num_iter += 1
            data_warp = data_warp.cuda()
            data_warp.requires_grad_()
            data_fix = data_fix.cuda()
            data_fix.requires_grad_()
            # data = torch.cat([data_warp, data_fix], dim=0)
            
            # forward
            # import pdb; pdb.set_trace()
            pred_warp, flow,_ = self.net(data_warp,data_fix)
           
            losses = self.criterion(pred_warp, data_fix, flow)
            if type(losses) is tuple or type(losses) is list:
                total_loss = None
                for i, loss in enumerate(losses):
                    loss = loss.mean()
                    if total_loss is None:
                        total_loss = loss
                    else:
                        total_loss += loss
                    # self.writer.add_scalar('loss%d' % i, loss.item(), self.num_iter)
                    print("loss%d: %.6f" % (i, loss.item()))
                loss = total_loss
            else:
                loss = losses.mean()

            print('epoch %d, iter %d/%d, loss: %f' % 
                    (self.num_epoch, i_batch, max_iter, loss.item()))

            # backward
            # if not math.isnan(loss.item()):
            #     loss.backward()
            # if i_batch % iter_size == 0:
            #     nn.utils.clip_grad_norm_(self.net.parameters(), 10)
            #     self.optimizer.step()
            #     self.optimizer.zero_grad()
            # torch.cuda.empty_cache()
            self.optimizer.zero_grad()   # reset gradient
            loss.backward()
            self.optimizer.step()            
            if self.num_epoch <= self.warm_epoch:
                self.warmup_scheduler.step()
        self.num_epoch += 1
        print(f"GPU memory usage:{torch.cuda.memory_allocated()/1024**2}MB")
        return loss.item()


    def save_model(self):
        model_name = 'epoch_%04d.pt' % (self.num_epoch-1)
        save_path = os.path.join(self.output_dir, model_name)
        save_model = self.net
        torch.save(save_model.state_dict(), save_path)
        return save_path

    def test_result(self):
        self.num_iter = 0
        # 模型测试
        model_name = 'epoch_%04d.pt' % (self.num_epoch-1)
        # model_name = 'epoch_0295.pt'
        save_path = os.path.join(self.output_dir, model_name)
        model = self.net
        model.load_state_dict(torch.load(save_path))
        # max_iter = len(self.batch_data)
        # print(max_iter)
        model.eval()
        # 在评估模式下进行测试或验证
        with torch.no_grad():
            fix_warp_ncc=0
            fix_move_ncc=0
            fix_warp_mse=0
            fix_move_mse=0
            fix_warp_psnr=0
            fix_move_psnr=0
            # fix_warp_hd=0
            # fix_move_hd=0
            # 保存图像的路径
            # save_dir_warp = os.path.join(self.output_dir, "warp")
            # save_dir_fix = os.path.join(self.output_dir, "fix")
            # save_dir_flow = os.path.join(self.output_dir, "flow")
            # save_dir_move = os.path.join(self.output_dir, "move")
            # os.makedirs(save_dir_warp, exist_ok=True)
            # os.makedirs(save_dir_fix, exist_ok=True)
            # os.makedirs(save_dir_flow, exist_ok=True)
            # os.makedirs(save_dir_move, exist_ok=True)
            for i_batch, (data_warp, data_fix) in enumerate(self.val_data):
                self.num_iter += 1
                data_warp = data_warp.cuda()
                data_warp.requires_grad_()
                data_fix = data_fix.cuda()
                data_fix.requires_grad_()
                # data = torch.cat([data_warp, data_fix], dim=0)

                pred_warp, flow,_ = self.net(data_warp,data_fix)
                flow = flow.permute(0, 2, 3, 4, 1)
                # cpu
                # pred_warp_np = pred_warp[0,0].cpu().numpy()
                # data_fix_np = data_fix[0,0].cpu().numpy()
                # data_flow_np = flow[0].cpu().numpy()
                # data_move_np = data_warp[0,0].cpu().numpy()
                # gpu
                # pred_warp_np = pred_warp[0,0]
                # data_fix_np = data_fix[0,0]
                # data_flow_np = flow[0]
                # data_move_np = data_warp[0,0]
                
                # pred_warp_itk=sitk.GetImageFromArray(pred_warp_np)
                # data_fix_itk=sitk.GetImageFromArray(data_fix_np)
                # data_flow_itk=sitk.GetImageFromArray(data_flow_np)    
                # data_move_itk=sitk.GetImageFromArray(data_move_np)  
                # sitk.WriteImage(pred_warp_itk, os.path.join(save_dir_warp, f'pred_warp_{i_batch}.nii'))
                # sitk.WriteImage(data_fix_itk, os.path.join(save_dir_fix, f'data_fix_{i_batch}.nii'))
                # sitk.WriteImage(data_flow_itk, os.path.join(save_dir_flow, f'data_flow_{i_batch}.nii'))
                # sitk.WriteImage(data_move_itk, os.path.join(save_dir_move, f'data_move_{i_batch}.nii'))
                
                fix_warp_ncc += 1.0-NCCLoss(window=(9,9,9))(data_fix,pred_warp)
                fix_move_ncc += 1.0-NCCLoss(window=(9,9,9))(data_warp,data_fix)
                fix_warp_mse += calculate_mse(data_fix,pred_warp)
                fix_move_mse += calculate_mse(data_warp,data_fix)
                fix_warp_psnr += pnsr(data_fix,pred_warp)
                fix_move_psnr += pnsr(data_warp,data_fix)
                # fix_warp_hd += hausdorff_distance(data_fix,pred_warp)
                # fix_move_hd += hausdorff_distance(data_warp,data_fix)

            print(f"epoch {self.num_epoch-1} fix_warp_ncc:{fix_warp_ncc/(self.num_iter)}  fix_move_ncc:{fix_move_ncc/(self.num_iter)}")
            print(f"epoch {self.num_epoch-1} fix_warp_mse:{fix_warp_mse/(self.num_iter)}  fix_move_mse:{fix_move_mse/(self.num_iter)}")
            print(f"epoch {self.num_epoch-1} fix_warp_psnr:{fix_warp_psnr/(self.num_iter)}  fix_move_psnr:{fix_move_psnr/(self.num_iter)}")
            # print(f"epoch {self.num_epoch-1} fix_warp_hd:{fix_warp_hd/(self.num_iter)}  fix_move_hd:{fix_move_hd/(self.num_iter)}")

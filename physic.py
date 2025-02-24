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
# from dataset_new.hospital_dataset import DIRLabDataset
# from dataset_new.pair_dataset import PairDataset

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

    def __init__(self, output_dir):
        val_data = DIRLabDataset('E:\dirlab',mode='val',inshape=128)
        val_data = PairDataset(val_data)
        print(len(val_data))
        self.dataset_val = val_data
        self.output_dir = output_dir

        self.val_data = DataLoader(self.dataset_val, batch_size=1, shuffle=False,
            num_workers=1, collate_fn=CollateFn(), pin_memory=True, drop_last=True)

        self.num_iter = 0


    # 配准函数
    def register(self, fixed_np, moving_np):
        # 将 numpy 数组转换为 SimpleITK 图像
        fixed_image = sitk.GetImageFromArray(fixed_np)
        moving_image = sitk.GetImageFromArray(moving_np)

        # 初始化配准器
        registration_method = sitk.ImageRegistrationMethod()

        # 设置配准方法为多分辨率配准（直接在注册方法中设置）
        registration_method.SetShrinkFactorsPerLevel([2, 2, 1])
        registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
        registration_method.SetSmoothingSigmasAreSpecifiedInPhysicalUnits(True)

        # 设置相似性度量（使用 AdvancedMattesMutualInformation）
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        # 设置优化器（穷举优化器）
        # optimizer = sitk.ExhaustiveOptimizer()
        # optimizer.SetNumberOfSteps([5, 5, 5])  # 每个维度的步数，可以根据需要调整
        # optimizer.SetStepLength(1.0)  # 步长，可以根据需要调整
        registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
        registration_method.SetOptimizerScalesFromPhysicalShift()

        # 设置插值器（线性插值）
        registration_method.SetInterpolator(sitk.sitkLinear)

        # 设置初始变换（B-spline 变换）
        bspline_transform = sitk.BSplineTransformInitializer(fixed_image, [4, 4, 4])
        registration_method.SetInitialTransform(bspline_transform)

        # 执行配准
        final_transform = registration_method.Execute(fixed_image, moving_image)

        # 应用变换到移动图像
        resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 0.0, moving_image.GetPixelID())

        # 计算形变场
        deformation_field_filter = sitk.TransformToDisplacementFieldFilter()
        deformation_field_filter.SetReferenceImage(fixed_image)
        deformation_field = deformation_field_filter.Execute(final_transform)

        # 将配准后的图像转换回 numpy 数组
        pred_warp = sitk.GetArrayFromImage(resampled_image)

        return pred_warp



    def test_result(self):
        self.num_iter = 0
        fix_warp_ncc=0
        fix_move_ncc=0
        fix_warp_mse=0
        fix_move_mse=0
        fix_warp_psnr=0
        fix_move_psnr=0
        # 保存图像的路径
        save_dir_warp = os.path.join(self.output_dir, "warp")
        # save_dir_fix = os.path.join(self.output_dir, "fix")
        # save_dir_flow = os.path.join(self.output_dir, "flow")
        # save_dir_move = os.path.join(self.output_dir, "move")
        os.makedirs(save_dir_warp, exist_ok=True)
        # os.makedirs(save_dir_fix, exist_ok=True)
        # os.makedirs(save_dir_flow, exist_ok=True)
        # os.makedirs(save_dir_move, exist_ok=True)
        for i_batch, (data_warp, data_fix) in enumerate(self.val_data):
            self.num_iter += 1
            print(self.num_iter)
            # data_warp = data_warp.cuda()
            # data_warp.requires_grad_()
            # data_fix = data_fix.cuda()
            # data_fix.requires_grad_()
            # data = torch.cat([data_warp, data_fix], dim=0)
            pred_warp_np = data_warp[0,0].cpu().numpy()
            data_fix_np = data_fix[0,0].cpu().numpy()
            pred_warp = self.register(data_fix_np,pred_warp_np)
            pred_warp = pred_warp[np.newaxis, np.newaxis, ...]
            pred_warp = torch.from_numpy(pred_warp)
            # flow = flow.permute(0, 2, 3, 4, 1)
            if self.num_iter<10:
                # cpu
                # pred_warp_np = pred_warp[0,0].cpu().numpy()
                # data_fix_np = data_fix[0,0].cpu().numpy()
                # data_flow_np = flow[0].cpu().numpy()
                # data_move_np = data_warp[0,0].cpu().numpy()
                # gpu
                pred_warp_np = pred_warp[0,0]
                # data_fix_np = data_fix[0,0]
                # data_flow_np = flow[0]
                # data_move_np = data_warp[0,0]
                
                pred_warp_itk=sitk.GetImageFromArray(pred_warp_np)
                # data_fix_itk=sitk.GetImageFromArray(data_fix_np)
                # data_flow_itk=sitk.GetImageFromArray(data_flow_np)    
                # data_move_itk=sitk.GetImageFromArray(data_move_np)  
                sitk.WriteImage(pred_warp_itk, os.path.join(save_dir_warp, f'pred_warp_{i_batch}.nii'))
                # sitk.WriteImage(data_fix_itk, os.path.join(save_dir_fix, f'data_fix_{i_batch}.nii'))
                # sitk.WriteImage(data_flow_itk, os.path.join(save_dir_flow, f'data_flow_{i_batch}.nii'))
                # sitk.WriteImage(data_move_itk, os.path.join(save_dir_move, f'data_move_{i_batch}.nii'))
            
            fix_warp_ncc += 1.0-NCCLoss(window=(9,9,9))(data_fix,pred_warp)
            print('fix_warp_ncc',fix_warp_ncc)
            fix_move_ncc += 1.0-NCCLoss(window=(9,9,9))(data_warp,data_fix)
            fix_warp_mse += calculate_mse(data_fix,pred_warp)
            fix_move_mse += calculate_mse(data_warp,data_fix)
            fix_warp_psnr += pnsr(data_fix,pred_warp)
            fix_move_psnr += pnsr(data_warp,data_fix)

        print(f"epoch 90 fix_warp_ncc:{fix_warp_ncc/(self.num_iter)}  fix_move_ncc:{fix_move_ncc/(self.num_iter)}")
        print(f"epoch 90 fix_warp_mse:{fix_warp_mse/(self.num_iter)}  fix_move_mse:{fix_move_mse/(self.num_iter)}")
        print(f"epoch 90 fix_warp_psnr:{fix_warp_psnr/(self.num_iter)}  fix_move_psnr:{fix_move_psnr/(self.num_iter)}")

if __name__ == '__main__':
    os.makedirs('./output', exist_ok=True)
    output_dir = './output/%s' % ("p_dirlab")
    os.makedirs(os.path.join(output_dir), exist_ok=True)
    solver = Solver(output_dir)
    solver.test_result()
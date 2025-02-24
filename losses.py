import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NCCLoss:

    def __init__(self, window=(9, 9, 9)):
        self.win = window
     
    def __call__(self, I, J, weight=False):
        sum_weight = torch.ones((1, 1, self.win[0], self.win[1], self.win[2]),
            device=I.device, dtype=torch.float)

        I2 = I*I
        J2 = J*J
        IJ = I*J

        # compute local sum
        padding = [ int((k)/2) for k in self.win ]
        I_sum = F.conv3d(I, sum_weight, padding=padding)
        J_sum = F.conv3d(J, sum_weight, padding=padding)
        I2_sum = F.conv3d(I2, sum_weight, padding=padding)
        J2_sum = F.conv3d(J2, sum_weight, padding=padding)
        IJ_sum = F.conv3d(IJ, sum_weight, padding=padding)

        # cross correlation
        win_size = self.win[0] * self.win[1] * self.win[2]
        I_u = I_sum/win_size
        J_u = J_sum/win_size


        cross = IJ_sum - J_u*I_sum - I_u*J_sum + I_u*J_u*win_size
        I_var = I2_sum - 2*I_u*I_sum + I_u*I_u*win_size
        J_var = J2_sum - 2*J_u*J_sum + J_u*J_u*win_size

        cc = cross*cross / (I_var*J_var + 1e-5)

        #weight in different slice
        if weight:
            cc = torch.mean(cc, dim=[0,1,3,4])
            w = torch.zeros(cc.size(0), dtype=cc.dtype, device = cc.device)
            for i in range(0, w.size(0), 4):
                w[i] = 1.0/w.size(0)*4.0
            cc = cc*w

        return 1.0-cc.mean()


def grad_loss3d(x):

    diff_d = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
    diff_h = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
    diff_w = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

    loss = (torch.mean(diff_d*diff_d)
        + torch.mean(diff_h*diff_h)
        + torch.mean(diff_w*diff_w)) / 3.0

    return loss

def dice_loss_bi(y_true, y_pred):
    """
    N-D dice for segmentation
    """
    import pdb; pdb.set_trace()
    ndims = len(list(y_pred.size())) - 2
    vol_axes = list(range(2, ndims+2))
    print(y_pred.shape)
    top = 2 * (y_true * y_pred).sum(dim=vol_axes)
    bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom)
    return -dice

class DiceLossMultiClass(nn.Module):
    """Dice loss from two inputs of segmentation (between a mask and a probability map)"""

    def __init__(self, n_class=None, no_bg=False, softmax=False, eps=1e-7):
        super(DiceLossMultiClass, self).__init__()
        self.n_class = n_class
        self.eps = eps
        self.no_bg = no_bg
        self.softmax = softmax  # if the source inputs are in 0~1 range
        # self.source_one_hot = nn.Parameter()
        # self.target_one_hot = nn.Parameter()
    
    def mask_to_one_hot(self, mask, n_classes):
        """
        Convert a segmentation mask to one-hot coded tensor
        :param mask: mask tensor of size Bx1xDxHxW
        :param n_classes: number of classes
        :return: one_hot: BxCxDxHxW
        """

        one_hot_shape = list(mask.shape)
        one_hot_shape[1] = n_classes

        mask_one_hot = torch.zeros(one_hot_shape).to(mask.device)

        mask_one_hot.scatter_(1, mask.long(), 1)

        return mask_one_hot


    def forward(self, source, target):
        """
        :param source: Tensor of size Bx1xDxHxW, warpped label range to (0~n_class)
        :param target: Tensor of size Bx1xDxHxW, ground truth range to (0~n_class)
        :return:
        """
        assert source.shape[0] == target.shape[0]
        assert source.shape[-3:] == target.squeeze().shape[-3:]

        if self.n_class is None:
            self.n_class = max(torch.unique(target).max(), torch.unique(source).max()).long().item() + 1

        shape = list(source.shape)

        if self.softmax:
            source = F.softmax(source, dim=1)

        # flat the spatial dimensions and transform it into one-hot coding
        if target.shape[1] == source.shape[1]:
            target_flat = self.mask_to_one_hot(target.view(shape[0], 1, -1), self.n_class)
            source_flat = self.mask_to_one_hot(source.view(shape[0], 1, -1), self.n_class)
        else:
            target_flat = None
            raise ValueError("Incorrect size of target tensor: {}, should be {} or []".format(target.shape, shape))

        # does not consider background
        if self.no_bg:
            source_flat = source_flat[:, 1:, :]
            target_flat = target_flat[:, 1:, :]

        #
        source_volume = source_flat.sum(2)
        target_volume = target_flat.sum(2)

        intersection = (source_flat * target_flat).sum(2)
        scores = (2. * (intersection.float()) + self.eps) / (
                (source_volume.float() + target_volume.float()) + 2 * self.eps)
        return - scores.sum()/scores.shape[1]

def calculate_nmi(fixed_image, warped_image):
    """
    计算标准化互信息（NMI）
    """
    # 确保图像的像素值和为1
    fixed_image = fixed_image / torch.sum(fixed_image, dim=(1, 2, 3, 4), keepdim=True)
    warped_image = warped_image / torch.sum(warped_image, dim=(1, 2, 3, 4), keepdim=True)
    
    # 计算联合概率分布
    joint_prob = fixed_image * warped_image
    
    # 计算边缘概率分布
    fixed_prob = torch.sum(fixed_image, dim=(2, 3, 4), keepdim=True)
    warped_prob = torch.sum(warped_image, dim=(2, 3, 4), keepdim=True)
    
    # 计算互信息（MI）
    mi = torch.sum(joint_prob * torch.log(joint_prob / (fixed_prob * warped_prob + 1e-6)))
    
    # 计算熵
    fixed_entropy = -torch.sum(fixed_image * torch.log(fixed_image + 1e-6), dim=(1, 2, 3, 4))
    warped_entropy = -torch.sum(warped_image * torch.log(warped_image + 1e-6), dim=(1, 2, 3, 4))
    
    # 计算标准化互信息（NMI）
    nmi = mi / torch.sqrt(fixed_entropy * warped_entropy)
    
    return nmi

def calculate_mse(fixed_image, warped_image):
    """
    计算均方差（MSE）
    """
    return torch.mean((fixed_image - warped_image) ** 2)

def dice_coefficient(fixed_image, warped_image):
    """
    计算Dice系数（DSC）
    """
    fixed_image = fixed_image.flatten()
    warped_image = warped_image.flatten()
    intersection = (fixed_image * warped_image).sum()
    return (2 * intersection) / (fixed_image.sum() + warped_image.sum())

from scipy.spatial.distance import cdist

def hausdorff_distance(fixed_image, warped_image):
    """
    计算HD95。
    
    参数:
    fixed_image -- 大小为[1,1,128,128,128]的固定图像tensor。
    moving_image -- 大小为[1,1,128,128,128]的移动图像tensor。
    warped_image -- 大小为[1,1,128,128,128]的变形图像tensor。
    
    返回:
    hd95 -- HD95值。
    """
    # 将图像转换为numpy数组
    fixed = fixed_image.cpu().numpy()
    warped = warped_image.cpu().numpy()
    
    # 计算固定图像和变形图像之间的距离矩阵
    distances = cdist(fixed, warped, metric='euclidean')
    
    # 找到固定图像中每个体素到变形图像最近点的距离
    nearest_distances = np.min(distances, axis=1)
    
    # 计算HD95
    sorted_distances = np.sort(nearest_distances)
    hd95 = np.percentile(sorted_distances, 95)
    
    return hd95


def pnsr(original_image, processed_image):
        # 确保输入张量的数据类型为浮点数
        original_image = original_image.float()
        processed_image = processed_image.float()

        # 计算均方误差（MSE）
        mse = torch.mean((original_image - processed_image) ** 2)

        # 如果MSE为0，PSNR无穷大，这里我们使用一个很小的非零值替代0以避免错误
        if mse == 0:
            return float('inf')

        # 计算PSNR
        max_intensity = 1.0  # 假设图像像素值的范围在0到255之间
        psnr = 10 * torch.log10((max_intensity ** 2) / mse)

        return psnr

if __name__ == "__main__":

    I = torch.rand(1, 1, 10, 10, 10)
    J = torch.rand(1, 1, 10, 10, 10)

    ncc_loss = NCCLoss(window=(3,3,3))

    print( ncc_loss(I, J) )

    

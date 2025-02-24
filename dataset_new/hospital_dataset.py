# import cv2
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
import os
from scipy.ndimage import zoom
class DIRLabDataset(Dataset):

    def __init__(self, data_root='./xzp/pancreas4d_anonymization_cut', mode='train',inshape=128):
        if mode=='train':
            data_idx = [3,7,8,10,13,14,20,22,24]#[2,3,4,5,6,7,8,10,12,13,14,18,19,20,22,23,24,25,27,28,30]
        if mode=='val':
            data_idx = [1]
        data_list = [ ("case%d_4d" % (i) , "case%d_T%02d" % (i,j)) for i in data_idx for j in range(0, 100,10)]
        # data_list = [ (i , j) for i in data_idx for j in range(0, 100,10)]
        data_list.sort()
        self.data_root = data_root
        self.data_list = data_list 
        self.inshape = inshape

        # self.data = []
        # for i in range(len(data_list)):
        #     id, time = self.data_list[i]
        #     data_id = "Case%dPack" % (id)
        #     data_time = "case%d_T%02d" % (id,time)
        #     data_path = os.path.join(self.data_root, data_id,'Images_change', "%s.nii" % (data_time))
        #     data = sitk.ReadImage(data_path)
        #     data = sitk.GetArrayFromImage(data)
        #     self.data_shape = data.shape
        #     # print('raw shape', self.data_shape)
        #     # normalization
        #     data = data.astype(np.float32)
        #     data = (data-data.min())/(data.max()-data.min())
        #     resized_img = zoom(data, [self.inshape/self.data_shape[0]] * 3, order=1)
        #     self.data.append(resized_img)

    def __len__(self):
        return len(self.data_list)
        # return len(self.data)


    def __getitem__(self, index):
        data_id, data_time = self.data_list[index]
        data_path = os.path.join(self.data_root, data_id, "%s.nii" % (data_time))
        data = sitk.ReadImage(data_path)
        data = sitk.GetArrayFromImage(data)
        self.data_shape = data.shape
        data = data.astype(np.float32)
        data = (data-data.min())/(data.max()-data.min())
        resized_img = zoom(data, [self.inshape/self.data_shape[0]] * 3, order=1)
        return resized_img

        # return self.data[index]


if __name__ == '__main__':
    data_root='./xzp/dirlab'
    dataset = DIRLabDataset(data_root,mode='val',inshape=16)
    num_data = len(dataset)
    for i in range(num_data):
        data = dataset[i]
        print(data.shape)
        # for j in range(data.shape[0]):
        #     frame = data[j, :, :]
        #     frame = (255*frame).astype(np.uint8)
        #     print(frame.shape)
            # cv2.imshow("frame", frame)
            # cv2.waitKey()

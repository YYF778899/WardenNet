import cv2
import numpy as np
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from scipy.spatial import distance
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
import scipy.ndimage
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm
from WardenNet import WardenNet
from skimage.filters import threshold_otsu
from scipy.ndimage import median_filter


def DataPreProcess(label_path,sample_path):
    sampleList = os.listdir(sample_path)
    sampleData = []
    for sample in sampleList:
        sampleData.append((os.path.join(sample_path,sample),
                          os.path.join(label_path,sample)))
    return sampleData

class myDataset(Dataset):
    def __init__(self, datalist, transform):

        self.datalist = datalist
        self.transform = transform

    def min_max_normalize(self, image):
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            x_min = np.min(image[i])
            x_max = np.max(image[i])
            normalized[i] = (image[i] - x_min) / (x_max - x_min)+0.0000000000000001
        return normalized

    def normalize(self,image):
        normalized = np.zeros_like(image, dtype=np.float32)
        for i in range(image.shape[0]):
            mean = np.mean(image[i])
            var = np.mean(np.square(image[i] - mean))
            normalized[i] = (image[i] - mean) / np.sqrt(var)
        return normalized


    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        img_path, label_path  = self.datalist[idx]
        if self.transform == 'test':
            img_array = np.load(img_path,allow_pickle=True)
            label_array = np.load(label_path)
            img_id = (img_path.split('/')[-1]).split('.')[0]
            img_array = self.min_max_normalize(img_array)
            label_array = np.expand_dims(label_array, axis=0)
            return img_array.astype(np.float32), label_array.astype(np.float32),img_id

def min_max_normalize(image):
    x_min = np.min(image)
    x_max = np.max(image)
    return (image - x_min) / (x_max - x_min)


def Test(testData,weight_path):
    save_img_path = '/TestImage'

    myTestdata = myDataset(datalist=testData, transform='test')
    Test_dataloader = DataLoader(myTestdata, batch_size=1, shuffle=False, num_workers=1)

    model = WardenNet()
    pre_weight = torch.load(weight_path)
    model.load_state_dict({k.replace('module.', ''): v for k, v in pre_weight.items()})

    with torch.no_grad():
        model.eval()
        for testimg, testlabel, test_img_id in tqdm(Test_dataloader):

            test_img_array = testimg
            test_lb_array = testlabel
            test_output = model(test_img_array)

            ###--------------Save---------------------
            for test_i in range(len(test_output)):
                final_test_img = test_output[test_i,0].astype(np.float32)
                final_test_img = min_max_normalize(final_test_img)
                np.save(os.path.join(save_img_path,'TestNpy',test_img_id[test_i]+'.npy'),final_test_img)




if __name__ == '__main__':

    test_label_path = '/.../Label'
    test_sample_path = '/.../TestNpy'

    testData = DataPreProcess(test_label_path, test_sample_path)
    weight_path = '/.../val_loss_best_model.pth'
    Test(testData,weight_path)






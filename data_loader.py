import cv2
import numpy as np
import torch.utils.data as data
from PIL import Image
import os
from torchvision import transforms
import pywt
from pytorch_wavelets import DWTForward, DWTInverse


class train_data(data.Dataset):

    def __init__(self,train_path,label_path):
        self.train_path = train_path
        self.label_path = label_path
        self.data = []
        file_list = os.listdir(self.label_path)
        for i in file_list:
            label_name = os.path.join(self.label_path,i)
            data_name = os.path.join(self.train_path,i)
            self.data.append([data_name, label_name])

        self.trans = transforms.Compose([

            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        data_path, label_path = self.data[index]

        data = Image.open(data_path)
        label = Image.open(label_path)

        data.convert("YCbCr")
        label.convert("YCbCr")

        data = self.trans(data)
        label = self.trans(label)


        return data,label

    def __len__(self):
        return len(os.listdir(self.train_path))


class train_data_haar(data.Dataset):

    def __init__(self,train_path,label_path):
        self.DWT = DWTForward()
        self.IWT = DWTInverse()

        self.train_path = train_path
        self.label_path = label_path
        self.data = []
        file_list = os.listdir(self.label_path)
        for i in file_list:
            label_name = os.path.join(self.label_path,i)
            data_name = os.path.join(self.train_path,i)
            self.data.append([data_name, label_name])

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):

        data_path, label_path = self.data[index]

        data = Image.open(data_path)
        label = Image.open(label_path)

        data = np.array(data,dtype='float32')
        label = np.array(label,dtype='float32')

        data_ycbcr = cv2.cvtColor(data, cv2.COLOR_BGR2YCrCb)
        label_ycbcr = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

        # Get Y channel
        data = data_ycbcr[:, :, 0]
        label = label_ycbcr[:, :, 0]

        # haar wavelet transform
        dataLF, (dataHF1,dataHF2,dataHF3) = pywt.dwt2(data/255, 'haar')
        labelLF, (labelHF1,labelHF2,labelHF3)= pywt.dwt2(label/255, 'haar')
        data_wt = np.stack((dataLF,dataHF1,dataHF2,dataHF3),axis=-1)
        label_wt = np.stack((labelLF,labelHF1,labelHF2,labelHF3),axis=-1)

        data = self.trans(data/255)
        label = self.trans(label/255)

        data_wt = self.trans(data_wt)
        label_wt = self.trans(label_wt)

        return data,label,data_wt,label_wt

    def __len__(self):
        return len(os.listdir(self.train_path))


class test_data(data.Dataset):

    def __init__(self, test_path, label_path):
        self.test_path = test_path
        self.label_path = label_path
        self.data = []
        test_file_list = os.listdir(self.test_path)
        label_path_list = os.listdir(self.label_path)
        for i in range(len(label_path_list)):
            data_name = os.path.join(self.test_path, test_file_list[i])
            label_name = os.path.join(self.label_path, label_path_list[i])
            self.data.append([data_name, label_name])

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        data_path, label_path = self.data[index]

        img = Image.open(data_path)
        data = Image.open(data_path)
        label = Image.open(label_path)

        img = self.trans(img)
        data = self.trans(data)
        label = self.trans(label)
        name = data_path

        return img,data,label,name

    def __len__(self):
        return len(os.listdir(self.test_path))


class test_data_haar(data.Dataset):

    def __init__(self, test_path, label_path):
        self.test_path = test_path
        self.label_path = label_path
        self.data = []
        test_file_list = os.listdir(self.test_path)
        label_path_list = os.listdir(self.label_path)
        for i in range(len(label_path_list)):
            data_name = os.path.join(self.test_path, test_file_list[i])
            label_name = os.path.join(self.label_path, label_path_list[i])
            self.data.append([data_name, label_name])

        self.trans = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        data_path, label_path = self.data[index]


        data = Image.open(data_path)
        label = Image.open(label_path)

        data = np.array(data,dtype='float32')
        label = np.array(label,dtype='float32')

        data_ycbcr = cv2.cvtColor(data, cv2.COLOR_BGR2YCrCb)
        label_ycbcr = cv2.cvtColor(label, cv2.COLOR_BGR2YCrCb)

        data = data_ycbcr[:, :, 0]/255
        label = label_ycbcr[:, :, 0]/255

        dataCb = data_ycbcr[:, :, 1]
        dataCr = data_ycbcr[:, :, 2]

        data = self.trans(data)
        label = self.trans(label)

        name = data_path

        return data,label,name,dataCb,dataCr

    def __len__(self):
        return len(os.listdir(self.test_path))
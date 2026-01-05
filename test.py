from data_loader import test_data_haar
import argparse, os
import torch
from torch.utils.data import DataLoader
import numpy as np
import cv2
from model.InvJND import InvJND_Net
import torch.nn.functional as F

parser = argparse.ArgumentParser(description="PyTorch DRRN Test")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--cuda", default="true", help="Use cuda?")
opt = parser.parse_args()

#Load the model.
model1 = InvJND_Net()
model1.load_state_dict(torch.load('./Weights/checkpoint.pth'))
model1.cuda()

# get the test path
test_set = test_data_haar(test_path="./example/data"
                     ,label_path="./example/data")

test_data_loader = DataLoader(dataset=test_set, batch_size=opt.batchSize,shuffle=False)

k=1
metric_rmse = []
for data in test_data_loader:
    img,label,name,Cb,Cr = data
    input_img = img.cuda()
    LF,HF = model1.forward(input_img)
    CPL = model1.reverse(LF,HF)
    CPL = F.interpolate(CPL, size=(input_img.shape[2],input_img.shape[3]), mode='bicubic')
    Cpl = CPL.detach()

    name = str(name)
    name = name.split(".")[-2]
    name = name.split("\\")[-1]

    Cpl = Cpl.cpu()
    Cpl = Cpl.squeeze(0)
    Cpl = np.array(Cpl)
    Cpl = np.float32(Cpl)

    img = img.cpu()
    img = img.squeeze(0)
    img=np.array(img)
    img=np.float32(img)

    label = label.cpu()
    label = label.squeeze(0)
    label=np.array(label)
    label=np.float32(label)

    CPL = Cpl*255
    img = img * 255
    label = label * 255

    Cb = np.array(Cb)
    Cb = np.float32(Cb)

    Cr = np.array(Cr)
    Cr = np.float32(Cr)

    CPL_YCbCr = np.stack((Cpl*255,Cb,Cr),axis=-1)
    CPL_YCbCr = CPL_YCbCr.squeeze(0)

    CPL_RGB = cv2.cvtColor(CPL_YCbCr, cv2.COLOR_YCrCb2RGB)
    cv2.imwrite(f"./result/{name}_CPL.png", CPL_RGB)#Save CPL image.

    JND = np.abs(CPL - img)
    cv2.imwrite(f"./result/{name}_JND.png",JND.squeeze(0))#Save JND image.


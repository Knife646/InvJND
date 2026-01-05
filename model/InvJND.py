import torch
import torch.nn as nn
from pytorch_wavelets import DWTForward, DWTInverse


# DenseBlock定义
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self._make_layer(in_channels + i * growth_rate, growth_rate))

    def _make_layer(self, in_channels, growth_rate):
        layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.ReLU()
        )
        return layer

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            out = layer(torch.cat(features, dim=1))  # 在通道维度上拼接
            features.append(out)
        return torch.cat(features, dim=1)


# Residual Dense Block定义
class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channels,out_channels, growth_rate, num_layers):
        super(ResidualDenseBlock, self).__init__()
        self.dense_block = DenseBlock(in_channels, growth_rate, num_layers)
        # 1x1卷积用于调整通道数，使残差连接的输入与输出通道数一致
        self.conv1x1 = nn.Conv2d(in_channels + num_layers * growth_rate ,out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        # 通过DenseBlock
        dense_out = self.dense_block(x)
        # 通过1x1卷积调整通道数
        dense_out = self.conv1x1(dense_out)
        # 残差连接：输入 + DenseBlock的输出
        out = x + dense_out
        return out

class DWT(nn.Module):
    def __init__(self,J=1, wave='haar', mode='zero'):
        super(DWT, self).__init__()
        self.DWT = DWTForward(J=J, wave=wave, mode=mode)


    def forward(self,x):
        x1,[x2] = self.DWT(x)
        x2 = torch.squeeze(x2,dim=1)
        return x1,x2

class IWT(nn.Module):
    def __init__(self,wave='haar', mode='zero'):
        super(IWT, self).__init__()
        self.IWT = DWTInverse(wave=wave,mode=mode)

    def forward(self,x1,x2):
        x2 = [torch.unsqueeze(x2,dim=1)]
        x = self.IWT((x1,x2))
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels,middle_channels, out_channels, kernel_size=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, middle_channels, kernel_size=kernel_size, padding=padding)
        self.conv1 = nn.Conv2d(middle_channels, out_channels, kernel_size=kernel_size, padding=padding)

        self.bn = nn.BatchNorm2d(middle_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.relu(self.conv(x))
        x = self.relu(self.conv1(x))
        return x

class CascadConvBlock(nn.Module):
    def __init__(self, in_channels,middle_channels, out_channels):
        super(CascadConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, middle_channels, kernel_size=1, padding=0)
        self.conv1 = nn.Conv2d(middle_channels,middle_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(middle_channels)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv(x))
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x



class InvertibleBlock(nn.Module):
    def __init__(self,LF_channels,Mid_Channels,HF_channels,kernel_size=1,padding = 0):
        super(InvertibleBlock, self).__init__()
        self.nn1 = ConvBlock(in_channels=LF_channels,middle_channels=Mid_Channels,out_channels=HF_channels,kernel_size=kernel_size,padding=padding)
        self.nn2 = ConvBlock(in_channels=HF_channels, middle_channels=Mid_Channels, out_channels=LF_channels,
                             kernel_size=kernel_size, padding=padding)

        self.nn11 = ConvBlock(in_channels=LF_channels, middle_channels=Mid_Channels, out_channels=HF_channels,
                             kernel_size=kernel_size, padding=padding)
        self.nn22 = ConvBlock(in_channels=HF_channels, middle_channels=Mid_Channels, out_channels=LF_channels,
                             kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2):
        y1 = x1 * torch.exp(self.nn22(x2)) + self.nn2(x2)
        # y1 = x1 + self.nn2(x2)
        mid = torch.sigmoid(y1) * 2 - 1
        y2 = x2 * torch.exp(self.nn11(mid)) + self.nn1(y1)
        # # y2 = x2  + self.nn1(y1)
        return y1, y2

    def inverse(self, y1, y2):
        x2 = (y2 - self.nn1(y1)).div(torch.exp(self.nn11(y1)))
        # x2 = y2 - self.nn1(y1)
        mid = torch.sigmoid(x2) * 2 - 1
        x1 = (y1 - self.nn2(x2)).div(torch.exp(self.nn22(mid)))
        # x1 = y1 - self.nn2(x2)
        return x1, x2

class InvJND_Net(nn.Module):
    def __init__(self):
        super(InvJND_Net, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()
        self.blocks = nn.ModuleList([InvertibleBlock(1,64,3) for _ in range(4)])
        self.RDN = ResidualDenseBlock(1,1,16,4)
        self.Rec = ResidualDenseBlock(1,1,16,4)
        self.HFLearning =CascadConvBlock(3,128,3)

    def forward(self,x):
        x = self.RDN(x)
        x1,x2 = self.DWT(x)

        for block in self.blocks:
            x1, x2 = block(x1, x2)

        LF, HF = x1, x2
        HF_ = self.HFLearning(HF)

        return LF,HF_

    def reverse(self,LF, HF_):


        y1, y2 = LF, HF_

        for block in reversed(self.blocks):
            y1, y2 = block.inverse(y1, y2)

        Res = self.IWT(y1, y2)
        Res = self.Rec(Res)

        return Res


class InvJND_Net1(nn.Module):
    def __init__(self):
        super(InvJND_Net1, self).__init__()
        self.DWT = DWT()
        self.IWT = IWT()
        self.blocks = nn.ModuleList([InvertibleBlock(1,64,3) for _ in range(4)])
        self.RDN = ResidualDenseBlock(1,1,16,4)
        self.Rec = ResidualDenseBlock(1,1,16,4)
        self.HFLearning =CascadConvBlock(3,128,3)


    def forward(self,x):
        x = self.RDN(x)
        x1,x2 = self.DWT(x)

        for block in self.blocks:
            x1, x2 = block(x1, x2)

        LF, HF = x1, x2
        HF_ = self.HFLearning(HF)

        y1,y2 = LF,HF_

        for block in reversed(self.blocks):
            y1, y2 = block.inverse(y1, y2)

        Res = self.IWT(y1, y2)
        Res = self.Rec(Res)



        return Res










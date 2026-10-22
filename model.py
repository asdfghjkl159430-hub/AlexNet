import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        # AlexNet用ReLu作为激活函数
        self.ReLu = nn.ReLU()
        #定义要使用的网络
        # 第一层卷积层，输入通道3，输出通道96，卷积核大小11*11，padding=0，步长为4
        self.c1 = nn.Conv2d(in_channels=1,
                            out_channels=96,
                            kernel_size=11,
                            padding=0,
                            stride=4)
        # 第二层最大池化层，窗口大小3，步长为2
        self.s2 = nn.MaxPool2d(kernel_size=3,
                               stride=2)
        # 第三层卷积层，输入通道96，输出通道256，卷积核大小为5*5，padding=2，步长1
        self.c3 = nn.Conv2d(in_channels=96,
                            out_channels=256,
                            kernel_size=5,
                            padding=2,
                            stride=1)
        # 第四层最大池化层，窗口大小3*3，步长为2
        self.s4 = nn.MaxPool2d(3,2)
        # 第五层卷积层，输入通道256，输出通道384，卷积核大小3*3，padding=1，步长为1
        self.c5 = nn.Conv2d(256,384,3,1,1)
        # 第六层卷积层，输入通道384，输出通道384，卷积核大小3*3，padding1，步长1
        self.c6 = nn.Conv2d(384,384,3,1,1)
        # 第七层卷积层，输入通道384，输出通道256，padding=1，步长1
        self.c7 = nn.Conv2d(384,256,3,1,1)
        # 第八层最大池化层，窗口3，步长2
        self.s8 = nn.MaxPool2d(3,2)
        # 定义展平层
        self.flatten = nn.Flatten()
        # 第九层全连接层，输入9216，神经元4096
        self.f9 = nn.Linear(9216,4096)
        # 第十层全连接层，输入4096，神经元4096
        self.f10 = nn.Linear(4096,4096)
        # 第十一层全连接层，输入4096，输出10
        self.f11 = nn.Linear(4096,10)


    def forward(self,x):
        x = self.ReLu(self.c1(x))
        x = self.s2(x)
        x = self.ReLu(self.c3(x))
        x = self.s4(x)
        x = self.ReLu(self.c5(x))
        x = self.ReLu(self.c6(x))
        x = self.ReLu(self.c7(x))
        x = self.s8(x)
        x = self.flatten(x)

        x = self.ReLu(self.f9(x))
        x = F.dropout(x,0.5,training=self.training)
        x = self.ReLu(self.f10(x))
        x = F.dropout(x,0.5,training=self.training)
        x = self.f11(x)
        return x

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    summary(model,(1,227,227))

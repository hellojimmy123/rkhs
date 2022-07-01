import math

import torch
import torch.nn as nn
import torch.nn.init as init

from modules import *
import time



class model(nn.Module):
    def __init__(self, n1, n2, n3, k_size):
        super().__init__()
        wg0 = torch.zeros(n2, n1 * k_size[0] ** 3)
        wg1 = torch.randn(n3, n2 * k_size[1] ** 3)  # , torch.randn(n3,n4)
        init.kaiming_uniform_(wg0, a=math.sqrt(5))
        init.kaiming_uniform_(wg1, a=math.sqrt(5))
        # init.kaiming_uniform_(wg2, a=math.sqrt(5) )
        self.poly1 = Poly_Kernel(wg0, kernel_size=k_size[0])
        self.conv1 = nn.Conv3d(kernel_size=1, in_channels=n2, out_channels=n2)
        self.poly2 = Poly_Kernel(wg1, kernel_size=k_size[1])
        self.conv2 = nn.Conv3d(kernel_size=1, in_channels=n3, out_channels=n3)

    def forward(self, x):
        y = self.poly1(x)
        y = self.conv1(y)
        y = self.poly2(y)
        y = self.conv2(y)
        return y


test = model(n1=6, n2=10, n3=4, k_size=[3, 3])
cuda0 = torch.device("cuda:0")
test.to(cuda0)
for i in range(10) :
    c = torch.randn(30, 6, 40, 40, 40)
    c = c.to(cuda0)

    start = time.time()
    print(test(c).shape)
    end = time.time()
    print(end-start)


# poly = Poly_Kernel(wf,kernel_size=2)
# print( poly(c).shape )
# gauss = Gaussian_Kernel(wg, kernel_size=2)
# print(gauss(c).shape)

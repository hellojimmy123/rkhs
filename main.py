import math

import torch
import torch.nn as nn
import torch.nn.init as init

from modules import *
import time
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid
from s2cnn import s2_near_identity_grid



class model(nn.Module):
    def __init__(self, n1, n2, n3, k_size):
        super().__init__()
        f1 = 20
        f2 = 40
        f_output = 10
        b_in = 15  # 20 #30
        b_l1 = 10
        b_l2 = 6
        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.sconv1 = S2Convolution( nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.sconv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

       # self.out_layer = nn.Linear(f2, f_output)

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
        s0, s1, s2, s3, s4, s5 = x.shape
        # nimage, x, y, z, theta, phi
        y = self.sconv1( x.view(-1,1,s4,s5) )
        y = self.sconv2(y)
        y = y.view( s1, s2, s3, 1, )
        y = self.poly1(y)
        y = self.conv1(y)
        y = self.poly2(y)
        y = self.conv2(y)
        return y


test = model(n1=6, n2=10, n3=4, k_size=[3, 3])
cuda0 = torch.device("cuda:0")
test.to(cuda0)
# for i in range(10) :
#     c = torch.randn(30, 6, 40, 40, 40)
#     c = c.to(cuda0)
#
#     start = time.time()
#     print(test(c).shape)
#     end = time.time()
#     print(end-start)


# poly = Poly_Kernel(wf,kernel_size=2)
# print( poly(c).shape )
# gauss = Gaussian_Kernel(wg, kernel_size=2)
# print(gauss(c).shape)

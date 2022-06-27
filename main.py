import modules
from modules import *
import torch
import torch.nn as nn


c = torch.randn(3, 1, 4, 4, 4)
wf = torch.randn(10, 8)
wg = torch.randn(10, 8)

class model(nn.Module) :
    def __init__(self, wg ):
        super().__init__()
        self.poly = Poly_Kernel(wg, kernel_size=2)
        self.conv1 = nn.Conv3d(kernel_size=1,in_channels=10,out_channels = 20 )
    def forward(self, x):
        y = self.poly(x)
        y = self.conv1(y)
        return y



c = torch.randn(3, 1, 4, 4, 4)
wf = torch.randn(10, 8)
wg = torch.randn(10, 8)

test = model(wg=wg)
# poly = Poly_Kernel(wf,kernel_size=2)
# print( poly(c).shape )

print(test(c).shape)



# poly = Poly_Kernel(wf,kernel_size=2)
# print( poly(c).shape )
gauss = Gaussian_Kernel(wg, kernel_size=2)
# print(gauss(c).shape)
#
# conv3d = nn.Conv3d(in_channels = 10, out_channels=20, kernel_size = 1)
# print(conv3d(gauss(c)).shape)

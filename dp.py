import torch
import torch.nn as nn
import time 
from torch.nn import DataParallel as DP
from models import rk_conv
from sync_batchnorm.batchnorm import convert_model

class model(nn.Module) :
    def __init__(self) :
        super().__init__()
       # self.conv1 = nn.Conv3d(kernel_size=3,in_channels=1,out_channels=20)
        #self.conv2 = nn.Conv3d(kernel_size=3,in_channels=20,out_channels=40)
        self.bn1 = nn.BatchNorm3d( 1, affine=False ) 
    def forward(self,x) :
        print(x.shape) 
        return self.bn1(x)       
#  return #self.conv2(self.bn1(self.conv1(x))) 

ksize = [ 3, 3, 3 ] 
net = model().to(0)
dp_model = DP(net,device_ids=[0,1])#.cuda() 
dp_model = convert_model( dp_model ).cuda() 
#.to(0)
#x = torch.randn(1,1,100,100,100).to(0)
x = torch.cat( ( torch.zeros(1,1,1,1,1) +0.33456, torch.ones(3,1,1,1,1)-0.2),dim=0).to(0) 

 
x = x*1000
#x = torch.randn( 4, 1, 1, 1, 1 ).to(0)  

y = dp_model(x)

bn = nn.BatchNorm3d( 1, affine=False ).to(0)  
y1 = bn(x) 

print(y) 
print(y1) 
print( torch.max( torch.abs( y1 - y ) ) ) 

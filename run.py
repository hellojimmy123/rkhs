import random
import torch 
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import os 
import json
import numpy as np 
from models import *
from torch.nn import DataParallel as DP
from sync_batchnorm.batchnorm import convert_model
from threads import * 

class NET( nn.Module ) : 
    def __init__(self,  m2 ) : 
        super().__init__() 
       # self.m1 = m1 
        self.m2 = m2 
    def forward( self, x ) :  
        #y1 = x.view( 192, 1, -1 )  
      #  y1 = self.m1( x )
       # y1 = y1.view( -1, 48*32 ).transpose( 0, 1 )  
       # y1 = y1.view( 48*32, 1, 77, 95, 77).contiguous()  
        #y1 = torch.utils.checkpoint.checkpoint( self.m2, y1 )  
        y1 = self.m2( x ) 
        return y1  

gpu = torch.device('cuda:0')
cpu = torch.device('cpu') 
npoch = 40

laps = get_healpix_laplacians( 192, 2, 'combinatorial')
skernel = [ 4, 4, 4  ] 
kernel_size = [ 3, 3, 3 ] 
#n_channel = [ 64, 32, 16, 8, 1 ]
n_channel = [ 1, 16, 32  ] 

#s2 = s2_model( laps, n_channel, skernel ) 
#kernel_size )
#s2dp = DP( s2,device_ids=[0,1,2,3,4,5,6,7])
#s2dp = convert_model( s2dp ).cuda()

rk = rk_conv( kernel_size ).to(gpu)  
#rkdp = DP( rk,device_ids=[0,1,2,3,4,5,6,7])
#rkdp = convert_model( rkdp ).cuda()
 




net = NET(   rk ) 
#x = torch.randn( 192, 1, 77*95*77 ).to( gpu )
#x = torch.randn( 77*95*77, 192, 1 ) 
x = torch.randn( 1, 1536, 77, 95, 77 ).to(gpu)  

start_t = time.time()

for epoch in range(npoch):
    tot_loss = 0. 
    print( epoch ) 
    for i in range(280):
        y = net(x)
        print( y.shape ) 
    print( time.time() - start_t )  

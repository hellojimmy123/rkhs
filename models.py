import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.utils.checkpoint import checkpoint


from modules import *
import deepsphere
import time
import deepsphere
from deepsphere.models.spherical_unet.utils import SphericalChebBN
from deepsphere.models.spherical_unet.utils import SphericalChebBNPool
from deepsphere.utils.laplacian_funcs import get_healpix_laplacians
from deepsphere.layers.samplings.healpix_pool_unpool import HealpixAvgPool

def chunks( x ) : 
    with torch.no_grad() :
         mean = torch.mean(x)
         mean2 = torch.mean(x**2)
    print(mean,mean2) 

class s2_model(nn.Module):
    def __init__(self, laps, n_channel, kernel_size ):
        super().__init__()
        self.pooling_class = HealpixAvgPool()
        self.sconv1 = SphericalChebBN(n_channel[0], n_channel[1], laps[1], kernel_size[0], False )
        #self.sconv2 = SphericalChebBN(n_channel[1], n_channel[2], laps[1], kernel_size[1] ) 
        self.sconv2 = SphericalChebBNPool( n_channel[1], n_channel[2], laps[0], self.pooling_class, kernel_size[1], False )
        #self.sconv3 = SphericalChebBN( n_channel[2], n_channel[3], laps[0], kernel_size[2], False )
    def forward(self, x):
        y = self.sconv1( x )
        y = self.sconv2( y )
        #y = self.sconv3( y )
        #y = self.sconv4( y )
        #y2 = self.fc( y1.reshape( -1, 1536 ) ) 
        #y2 = checkpoint(self.sconv3, y1 )
        return y 

class linear_layer( nn.Module ) : 
    def __init__( self ) :  
        super().__init__() 
        self.fc1 = nn.Linear( 245760, 1024 )
        self.fc2 = nn.Linear( 1024, 128 )
        self.fc3 = nn.Linear( 128, 2 )
        self.ins1 = nn.InstanceNorm1d( 1024, affine = False)
        self.ins2 = nn.InstanceNorm1d( 128, affine = False ) 
    def forward(self,x) : 
        y = x.view(-1,245760)
        y = self.fc1( y ) 
        y = self.ins1( y ) 
        y = F.relu( y ) 
        y = self.fc2( y ) 
        y = self.ins2( y )
        y = F.relu( y )
        y = self.fc3( y ) 
        return y 




class rkhs(nn.Module) :
    def __init__(self,k_size):
        super().__init__()
        n0 = 1
        n1 = 16
        n2 = 32
        # f1 = 20
        # f2 = 30
        wg0 = torch.zeros(n1, n0 * k_size[0] ** 3)
        wg1 = torch.randn(n2, n1 * k_size[1] ** 3)  # , torch.randn(n3,n4)
        init.kaiming_uniform_(wg0, a=math.sqrt(5))
        init.kaiming_uniform_(wg1, a=math.sqrt(5))
        self.poly1 = Poly_Kernel(wg0, kernel_size=k_size[0])
        self.conv1 = nn.Conv3d(kernel_size=1, in_channels=n1, out_channels=n1)
        self.poly2 = Poly_Kernel(wg1, kernel_size=k_size[1])
        self.conv2 = nn.Conv3d(kernel_size=1, in_channels=n2, out_channels=n2)
        self.bn1 = nn.BatchNorm3d(n1)
        self.bn2 = nn.BatchNorm3d(n2)
    def forward(self, x ):
        s0, s1, s2, s3, s4 = x.shape
        y = self.poly1(x)
        y = self.conv1(y)
        z = checkpoint( self.poly2, y ) 
        z = self.conv2(z)
        return z
class cnn(nn.Module) : 
    def __init__(self,k_size):
        super().__init__()
        n0 = 1
        n1 = 16 
        n2 = 32 
        self.conv1 = nn.Conv3d( kernel_size=k_size[0], in_channels=n0, out_channels=n1)
        self.conv2 = nn.Conv3d( kernel_size=k_size[1], in_channels=n1, out_channels=n2)
        self.bn1 = nn.BatchNorm3d(n1) 
        self.bn2 = nn.BatchNorm3d(n2)
    def forward(self, x ):
        print(x.shape) 
        y = self.conv1(x)#.contiguous()                                                      
        y = self.bn1( y ) 
        z = self.conv2(y) 
       # z = self.bn2( z )                                                         
        return z

class rk_conv(nn.Module) : 
    def __init__(self,k_size):
        super().__init__()
        n0 = 1536 
        n1 = 200
        n2 = 80
        n3 = 1 
        #n4 = 64 
        self.conv1 = nn.Conv3d( kernel_size=k_size[0], in_channels=n0, out_channels=n1)
        self.conv2 = nn.Conv3d( kernel_size=k_size[1], in_channels=n1, out_channels=n2)
        self.conv3 = nn.Conv3d( kernel_size=k_size[2], in_channels=n2, out_channels=n3)
       # self.conv4 = nn.Conv3d( kernel_size=k_size[3], in_channels=n3, out_channels=n4)
        self.avg1 =  nn.AvgPool3d( kernel_size = 3)
        self.avg2 =  nn.AvgPool3d( kernel_size = 3) 
        self.avg3 = nn.AvgPool3d( kernel_size = 3 ) 
        self.conv11 = nn.Conv3d( kernel_size = 1, in_channels = n1, out_channels = n1 ) 
        self.conv22 = nn.Conv3d( kernel_size = 1, in_channels = n2, out_channels = n2 ) 
        self.conv33 = nn.Conv3d( kernel_size = 1, in_channels = n3, out_channels = n3)
      #  self.conv44 = nn.Conv3d( kernel_size = 1, in_channels = n4, out_channels = n4 )
        self.bn1 = nn.BatchNorm3d(n1) 
        self.bn2 = nn.BatchNorm3d(n2)
        self.bn3 = nn.BatchNorm3d(n3)
       # self.bn4 = nn.BatchNorm3d(n4)
  
    def forward(self, x ):
        y = self.conv1(x)
        y = (y+1)**2 
        y = self.conv11( y ) 
       # y = self.bn1( y )  
        z = self.conv2( y ) 
        z = ( z + 1 )**2 
        z = self.conv22( z )
      #  z = self.bn2( z )
        z = self.avg1( z )
        z = self.conv3( z ) 
        z = ( z + 1 )**2  
        z = self.conv33( z ) 
      #  z = self.bn3( z )    
      #  z = self.conv4( z ) 
      #  z = ( z + 1 )**2 
      #  z = self.conv44( z )                            
      #  z = self.bn4( z )
      #  z = self.avg2( z ) 
    
        return z





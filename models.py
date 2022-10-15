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
        self.sconv2 = SphericalChebBNPool( n_channel[1], n_channel[2], laps[0], self.pooling_class, kernel_size[1], False )
    def forward(self, x):
        y = self.sconv1( x )
        y = self.sconv2( y )
        return y.mean( dim = 1 )  




class rk_conv(nn.Module) : 
    def __init__(self,k_size):
        super().__init__()
        self.n0 = 64 
        self.n1 = 128
        self.n2 = 64
        self.n3 = 16 
        self.n4 = 16 
        self.conv1 = nn.Conv3d( kernel_size=k_size[0], in_channels=self.n0, out_channels=self.n1, padding = 'same' )
        self.conv2 = nn.Conv3d( kernel_size=k_size[1], in_channels=self.n1, out_channels=self.n2, padding = 'same' )
        self.conv3 = nn.Conv3d( kernel_size=k_size[2], in_channels=self.n2, out_channels=self.n3, padding = 'same' )
        self.conv4 = nn.Conv3d( kernel_size=k_size[3], in_channels=self.n3, out_channels=self.n4, padding = 'same' )
        self.max1 =  nn.MaxPool3d( kernel_size = 2)
        self.max2 =  nn.MaxPool3d( kernel_size = 2) 
        self.avg =  nn.AvgPool3d( kernel_size = 4, padding = 1)
        self.bn1 = nn.BatchNorm3d(1, affine = False) 
        self.bn2 = nn.BatchNorm3d(1, affine = False)
        self.bn3 = nn.BatchNorm3d(1, affine = False)
        self.bn4 = nn.BatchNorm3d(1, affine = False )
        self.fc1 = nn.Linear( 2400, 128 )
        self.fc2 = nn.Linear( 128, 8 ) 
  
    def forward(self, x ):
        y = self.conv1(x)
        y = self.bn1( y.view( self.n1, 1, 77, 95, 77) ).view( 1, self.n1, 77, 95, 77 )
        y = F.relu( y )  
        z = self.conv2( y ) 
        z = self.bn2( z.view( self.n2, 1, 77, 95, 77) ).view( 1, self.n2, 77, 95, 77 )
        z = F.relu( z )  
        z = self.max1( z )
        
        z = self.conv3( z )
        z = self.bn3( z.view( self.n3, 1, 38, 47, 38) ).view( 1, self.n3, 38, 47, 38 )
        z = F.relu( z )  
        z = self.max2( z ) 
        
        z = self.conv4( z )  
        z = self.bn4( z.view( self.n4, 1, 19, 23, 19) ).view( 1, self.n4, 19, 23, 19 )  
        z = F.relu( z )   
        z = self.avg( z )
         
        z = self.fc1( z.view(-1))
        z = F.relu( z ) 
        z = self.fc2( z.view(-1)) 
        z = F.relu( z ) 
 
        return z


class b0_net( nn.Module ) :
    def __init__( self, c=1.0, order=2, k_size = [ 3, 3, 3 ] ) :
        super().__init__()
        self.n0 = 1
        self.n1 = 16 
        self.n2 = 32
        self.n3 = 8 
        self.c = c
        self.order = order
        self.conv1 = nn.Conv3d( kernel_size=k_size[0], in_channels=self.n0, out_channels=self.n1, padding = 'same' )
        self.conv2 = nn.Conv3d( kernel_size=k_size[1], in_channels=self.n1, out_channels=self.n2, padding = 'same' ) 
        self.conv3 = nn.Conv3d( kernel_size=k_size[2], in_channels=self.n2, out_channels=self.n3, padding = 'same' )
        self.bn1 = nn.BatchNorm3d(1, affine = False)
        self.bn2 = nn.BatchNorm3d(1, affine = False)
        self.bn3 = nn.BatchNorm3d(1, affine = False)
        self.max1 =  nn.MaxPool3d( kernel_size = 2)
        self.max2 =  nn.MaxPool3d( kernel_size = 2)
        self.avg =  nn.AvgPool3d( kernel_size = 4, padding = 1 )
        self.fc1 = nn.Linear( 1200, 128 )
        self.fc2 = nn.Linear( 128, 2 ) 
    def forward( self, x ) : 
        y = self.conv1(x)
        y = ( y + self.c)**self.order
        y = self.bn1( y.view( self.n1, 1, 77, 95, 77) ).view( 1, self.n1, 77, 95, 77 )
        y = self.max1( y )

 
        z = self.conv2( y )
        z = ( z + self.c )**self.order
        z = self.bn2( z.view( self.n2, 1, 38, 47, 38) ).view( 1, self.n2, 38, 47, 38 )
        z = self.max2( z ) 

        z = self.conv3( z )
        z = ( z + self.c )**self.order 
        z = self.bn3( z.view( self.n3, 1, 19, 23, 19) ).view( 1, self.n3, 19, 23, 19 )
        z = self.avg( z )

        z = z.view(-1)
        z = self.fc1( z )
        z = F.relu( z )        
        z = self.fc2( z ) 
#        z = F.relu( z ) 
        return z 


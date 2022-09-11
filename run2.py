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
    def __init__(self, m1, m2, m3 ) : 
        super().__init__() 
        self.m1 = m1 
        self.m2 = m2 
        self.m3 = m3 
    def forward( self, x ) :  
        y = self.m1( x )
    # 192, 64, 16, 20, 16 
        y1 = y.view( 192, 64, -1 ) 
    # 192, 64, 5120
        y1 = y1.permute( 2, 0, 1 )
    # 5120, 192, 64
        y1 = self.m2( y1 )
    # 5120, 48, 1
        y2 = self.m3( y1 )
        return y2  


gpu = torch.device('cuda:0')
cpu = torch.device('cpu') 
npoch = 10

laps = get_healpix_laplacians( 192, 2, 'combinatorial')
kernel_size = [ 3, 3, 3, 3  ]
n_channel = [ 64, 32, 16, 8, 1 ]


s2 = s2_model( laps, n_channel, kernel_size )
s2dp = DP( s2,device_ids=[0,1,2,3,4,5,6,7])
s2dp = convert_model( s2dp ).cuda()
#x = torch.randn( 16000, 192, 256 ).to(gpu)
rk = rk_conv( kernel_size ) 
rkdp = DP( rk,device_ids=[0,1,2,3,4,5,6,7])
rkdp = convert_model( rkdp ).cuda() 
ll = linear_layer().to(gpu) 


net = NET(  rkdp, s2dp, ll ) 
#x = torch.randn( 192, 1, 77, 95, 77 ).to( gpu )

f = open('tmp.txt','w') 

with open('test.json','r') as gf :
    tmp_list = json.load( gf ) 

train_file_list = [] 
test_file_list = [] 


for i in tmp_list[0] :
    test_file_list.append(  (os.path.join('../../mix', str(i) + '.pt' ), os.path.join('../../mix', str(i) + '.label' )) )
for i in tmp_list[1] :
    train_file_list.append( ( os.path.join('../../mix', str(i) + '.pt' ), os.path.join('../../mix', str(i) + '.label' )) )

#tot_file_list = get_file_list() 
#train_file_list = tot_file_list[:280]
#test_file_list = tot_file_list[280:]

for item in test_file_list :
    tmp = item[0] 
    f.write(  tmp + '  ' )
f.write('\n')  

nts = non_torch_data_set( train_file_list, gpu, bf_size = 80 )
random.shuffle( train_file_list )
ntsb = non_torch_data_set( train_file_list, gpu, bf_size = 80 )
random.shuffle( train_file_list )

#tnts = non_torch_data_set( test_file_list, gpu, bf_size = 40 ) 
tds = no_cache_loading( test_file_list ) 

start_t = time.time()
loss_func = nn.CrossEntropyLoss()  
optimizer = optim.SGD( net.parameters(), lr = 5.0e-4, momentum = 0.9 ) 
#optimizer = optim.Adam( net.parameters(), lr = 1.0e-3 ) 
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,  gamma=0.99)


for epoch in range(npoch):
    f.write( 'epoch  {0:d}\n'.format(epoch)  ) 
    tot_loss = 0. 
    for i in range(280):
        optimizer.zero_grad() 
        f.write(  ' {0:d} '.format(i) ) 
       # print('\r', i, ' /280', end = ' ') 
        x, label = nts.get()
        label = label.unsqueeze(0).to(gpu)  
        #print( x.shape, x.device, label ) 
        y = net(x[:192])
       # print( 'running') 
        loss = loss_func( y, label ) 
        loss.backward()
        tot_loss += loss
        optimizer.step()  
        f.write( '{0:.10e}  {1:.10e} {2:.10e}  \n'.format(y[0,0],y[0,1], loss ) ) 
        #optimizer.step() 
        #print( time.time() - start_t )
        f.flush() 
    f.write('Loss: {0:.10e}\n\n\n'.format(tot_loss))
    scheduler.step()  
    nts.end_thread()
    nts = ntsb 
    if( epoch < npoch-2 ) : 
        ntsb = non_torch_data_set( train_file_list, gpu, bf_size = 100 )
        random.shuffle( train_file_list )
    
#nts.end_thread() 
    tot = len(tds)  
    cor = 0 
    with torch.no_grad() :
        for i in range(tot):
            x, label = tds[i]
            y = net( x[:192] )
            #loss = loss_func( y2, label ) 
            pred = y.argmax().to(cpu)  
            true_val = label.argmax()
            is_cor = pred.eq( true_val )
            f.write( ' {0:d} '.format(i) + str(is_cor) + '\n' ) 
            cor += is_cor 
        f.write( 'Accuracy : {0:.3f}'.format(cor/tot) + '\n' )
    f.flush() 
f.close() 

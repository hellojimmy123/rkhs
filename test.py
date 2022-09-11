

import numpy as np 
from threads import * 




tot_file_list = get_file_list() 
train_file_list = tot_file_list[:280]
test_file_list = tot_file_list[280:]

count = 0 
for i in train_file_list : 
    d, l = i 
    for j in test_file_list : 
        d0, l0 = j 
        if ( d == d0 )  :
            count += 1
print( count )  

from threads import *

file_list = get_file_list()
device = 'cpu'

nts = non_torch_data_set( file_list, device ) 


for i in range( 200) :
    tmp = nts.get()
nts.end_thread() 

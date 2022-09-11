import time
import os 
import random
import threading
import numpy 
import torch

def get_file_list(root='../../../mix',shuffle=True) : 
    file_list = [ (os.path.join(root,str(i)+'.pt'), os.path.join(root,str(i)+'.label')) for i in range(318)  ]  
    random.shuffle( file_list )  
    return file_list 




class Buffer:

    def __init__(self, size, device, file_list ):
        self.size = size
        self.buffer = []
        self.cur_pos = 0
        self.file_list = file_list 
        self.device = device 
        self.lock = threading.Lock()
        self.has_data = threading.Condition(self.lock)  # small sock depand on big sock
        self.has_pos = threading.Condition(self.lock)
          
    def get_size(self):
        return self.size

    def get(self):
        with self.has_data:
            while len(self.buffer) == 0:
                self.has_data.wait()
            result = self.buffer[0]
            del self.buffer[0]
            self.has_pos.notify_all()
        return result

    def put(self ):
        with self.has_pos:
            while ( len(self.buffer) >= self.size ):
                self.has_pos.wait()
            dataname, labelname = self.file_list[self.cur_pos] 
#os.path.join( data_folder, str(self.cur_pos) + '.npy' )
            #print("reading filename ", dataname )
            self.cur_pos += 1  
            self.buffer.append((torch.load(dataname),torch.load(labelname)))
            self.has_data.notify_all()

class buffer_loader() : 
    def __init__(self, buffer_size, file_list, device) : 
        self.buffer = Buffer(buffer_size, device,file_list)
        self.data_size = len(file_list) 
    def consumer(self):
        res = self.buffer.get()
        return res
    def producer(self):
        for _ in range(self.data_size):
            self.buffer.put()


class torch_data_set(torch.utils.data.Dataset) : 
    def __init__( self, file_list, device) : 
        self.loader = buffer_loader( 5, file_list, device )
        self.lens = len(file_list)
        self.th1 = threading.Thread(target=self.loader.producer) 
        self.th1.start()        
    def __getitem__(self,idx) :
        time1 = time.time() 
        res = self.loader.consumer()
        self.lens -= 1 
        print( time.time() - time1 ) 
        return res 
    def __len__(self) : 
        return self.lens 
    def end_thread(self) : 
        self.th1.join()

class non_torch_data_set() : 
    def __init__( self, file_list, device, bf_size = 10) :
        self.loader = buffer_loader( bf_size, file_list, device )
        self.th1 = threading.Thread(target=self.loader.producer)
        self.th1.start()
    def get(self) :
        res = self.loader.consumer()
        return res
    def end_thread(self) :
        self.th1.join()

def no_cache_loading( file_list ) : 
    res = []
    for i in file_list :
        dataname, labelname = i 
        res.append( (torch.load(dataname),torch.load(labelname) ) ) 
    return res  

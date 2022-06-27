import torch

a = torch.randn(5,1)
b = torch.randn(5,10)
#b = b.view(1,1,5,10)
a-b
#print(a.unsqueeze(3).shape)
# print( torch.unsqueeze(a,3) - b )
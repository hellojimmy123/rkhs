import torch
import torch.nn as nn
import s2cnn
from s2cnn import s2_near_identity_grid
from s2cnn import so3_near_identity_grid
from s2cnn import S2Convolution
from s2cnn import SO3Convolution
from s2cnn import so3_integrate
import torch.nn.functional as F


class s2net(nn.Module) :
    def __init__(self):
        super(s2net, self).__init__()

        f1 = 20
        f2 = 40
        f_output = 10

        b_in = 15  # 20 #30
        b_l1 = 10
        b_l2 = 6

        grid_s2 = s2_near_identity_grid()
        grid_so3 = so3_near_identity_grid()

        self.conv1 = S2Convolution(
            nfeature_in=1,
            nfeature_out=f1,
            b_in=b_in,
            b_out=b_l1,
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=f1,
            nfeature_out=f2,
            b_in=b_l1,
            b_out=b_l2,
            grid=grid_so3)

        self.out_layer = nn.Linear(f2, f_output)

    def forward(self, x):
        x = self.conv1(x)
        print("shape1: ", x.shape, end=" ")
        x = F.relu(x)
        x = self.conv2(x)
        print("shape2: ", x.shape, end=" ")
        x = F.relu(x)
        x = so3_integrate(x)
        print("shape3: ", x.shape, end=" ")
        x = self.out_layer(x)
        print("shape4: ", x.shape, end=" ")
        return x

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test1 = s2net().to(DEVICE)

tx = torch.randn(32,1,30,30,dtype=torch.float32).to(DEVICE)

print(test1(tx).shape)

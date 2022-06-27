# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import torch.nn as nn
import torch.nn.parameter
import unfoldNd


class Poly_Kernel(nn.Module):
    def __init__(self, w_init, kernel_size, dilation=1, stride=1, padding=0):
        super().__init__()
        self.w = nn.Parameter(w_init, requires_grad=True)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        s1, s2, s3, s4, s5 = x.shape
        y = unfoldNd.unfoldNd(x, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride, padding=0)
        y = (torch.einsum('ij,kjm->kim', self.w, y) + 1) ** 2
        return y.view(s1, -1, s3 - 1, s4 - 1, s5 - 1).contiguous()


class Gaussian_Kernel(nn.Module):
    def __init__(self, w_init, kernel_size, dilation=1, stride=1, padding=0):
        super().__init__()
        self.w = nn.Parameter(w_init, requires_grad=True)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        s1, s2, s3, s4, s5 = x.shape
        d1, d2 = self.w.shape
        y = unfoldNd.unfoldNd(x, kernel_size=self.kernel_size, dilation=self.dilation, stride=self.stride,
                              padding=0).transpose(1, 2)
        print(y.shape)
        y = (torch.unsqueeze(y, 2) - self.w.view(1, 1, d1, d2)) ** 2
        y = torch.exp(-torch.sum(y, dim=3))
        print(y.shape)
        return y.transpose(1, 2).view(s1, -1, s3 - 1, s4 - 1, s5 - 1).contiguous()


# def f(x, w):
#     s1, s2, s3, s4, s5 = x.shape
#     y = unfoldNd.unfoldNd(x, kernel_size=2, dilation=1, stride=1, padding=0)
#     y = (torch.einsum('ij,kjm->kim', w, y) + 1) ** 2
#     return y.view(s1, -1, s3 - 1, s4 - 1, s5 - 1).contiguous()
#
#
# def g(x, w):
#     s1, s2, s3, s4, s5 = x.shape
#     d1, d2 = w.shape
#     y = unfoldNd.unfoldNd(x, kernel_size=2, dilation=1, stride=1, padding=0).transpose(1, 2)
#     y = torch.unsqueeze(y, 3)
#     w = w.view(1, 1, d1, d2)
#     y = torch.exp(-(y - w) ** 2)
#     return y.permute(0, 2, 3, 1).view(s1, d1, d2, s3 - 1, s4 - 1, s5 - 1).contiguous()



# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


import torch
import unfoldNd
import torch.nn as nn


class poly(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, x, w):
        s1, s2, s3, s4, s5 = x.shape
        y = unfoldNd.unfoldNd(x, kernel_size=2, dilation=1, stride=1, padding=0)
        y = (torch.einsum('ij,kjm->kim', w, y) + 1) ** 2
        return y.view(s1, -1, s3 - 1, s4 - 1, s5 - 1).contiguous()


class Gaussian(nn.Module):
    def __init__(self):
        super.__init__()

    def forward(self, x, w):
        s1, s2, s3, s4, s5 = x.shape
        d1, d2 = w.shape
        y = unfoldNd.unfoldNd(x, kernel_size=2, dilation=1, stride=1, padding=0).transpose(1, 2)
        y = torch.unsqueeze(y, 3)
        w = w.view(1, 1, d1, d2)
        y = torch.exp(-(y - w) ** 2)
        return y.permute(0, 2, 3, 1).view(s1, d1, d2, s3 - 1, s4 - 1, s5 - 1).contiguous()


def f(x, w):
    s1, s2, s3, s4, s5 = x.shape
    y = unfoldNd.unfoldNd(x, kernel_size=2, dilation=1, stride=1, padding=0)
    y = (torch.einsum('ij,kjm->kim', w, y) + 1) ** 2
    return y.view(s1, -1, s3 - 1, s4 - 1, s5 - 1).contiguous()


def g(x, w):
    s1, s2, s3, s4, s5 = x.shape
    d1, d2 = w.shape
    y = unfoldNd.unfoldNd(x, kernel_size=2, dilation=1, stride=1, padding=0).transpose(1, 2)
    y = torch.unsqueeze(y, 3)
    w = w.view(1, 1, d1, d2)
    y = torch.exp(-(y - w) ** 2)
    return y.permute(0, 2, 3, 1).view(s1, d1, d2, s3 - 1, s4 - 1, s5 - 1).contiguous()


c = torch.randn(3, 1, 4, 4, 4)
wf = torch.randn(10, 8)
wg = torch.randn(8, 10)
print(g(c, wg).shape)

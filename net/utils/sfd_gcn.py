import torch
import torch.nn.functional as F
from torch import nn

class AttentionBlock(nn.Module):
    def __init__(self, alpha, in_channels=300):
        super().__init__()
        self.alpha = alpha
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fcn = nn.Sequential(
            nn.Linear(in_channels, in_channels//4),
            nn. BatchNorm1d(in_channels//4),
            nn.Sigmoid(),
            nn.Linear(in_channels//4, in_channels)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        N, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous()
        avg_pooling = self.pooling(x).view(N, T)
        fcn = self.fcn(avg_pooling).view(N,T)
        x1 = self.softmax(fcn)
        x3 = x * x1.view(N,T,1,1).expand_as(x)

        return (x + x3 * self.alpha).permute(0, 2, 1, 3).contiguous()
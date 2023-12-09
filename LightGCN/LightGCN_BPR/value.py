import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, opt):
        super(Value, self).__init__()
        self.dim = opt.user_dim
        self.value = opt.value
        if self.value == 'triplet':
            self.l1 = nn.Sequential(nn.Linear(self.dim, 256, bias=True), nn.ReLU(), nn.Linear(256, 256, bias=True), nn.ReLU(), nn.Linear(256, 128, bias=True), nn.ReLU(), nn.Linear(128, 1, bias=False))
            self.l2 = nn.Sequential(nn.Linear(self.dim * 2, 256, bias=True), nn.ReLU(), nn.Linear(256, 256, bias=True), nn.ReLU(), nn.Linear(256, 128, bias=True), nn.ReLU(), nn.Linear(128, 1, bias=False))
            self.l3 = nn.Sequential(nn.Linear(self.dim * 3, 256, bias=True), nn.ReLU(), nn.Linear(256, 256, bias=True), nn.ReLU(), nn.Linear(256, 128, bias=True), nn.ReLU(), nn.Linear(128, 1, bias=False))
            self.weight = nn.Parameter(torch.randn(3))
        elif self.value == 'overall':
            self.l1 = nn.Sequential(nn.Linear(self.dim * 3, 256, bias=True), nn.ReLU(), nn.Linear(256, 128, bias=True), nn.ReLU(), nn.Linear(128, 1, bias=False), nn.Sigmoid())
            for layer in self.l1.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, user, pos_item, neg_item):
        if self.value == 'overall':
            state = torch.concat((user, pos_item, neg_item), dim=1)
            res = self.l1(state)
        elif self.value == 'triplet':
            s1 = self.l1(user * pos_item - user * neg_item)
            s2 = self.l2(torch.concat((user * pos_item, user * neg_item), dim=1))
            s3 = self.l3(torch.concat((user, pos_item, neg_item), dim=1))
            weight = F.softmax(self.weight, dim=0)
            res = F.sigmoid(s1 * weight[0] + s2 * weight[1] + s3 * weight[2])
        return res
    


    
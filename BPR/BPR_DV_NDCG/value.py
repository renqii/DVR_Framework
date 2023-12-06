import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

class Value(nn.Module):
    def __init__(self, opt):
        super(Value, self).__init__()
        self.dim = opt.user_dim
        self.linear1 = nn.Linear(self.dim * 3, 256, bias=True)
        self.linear2 = nn.Linear(256, 128, bias=True)
        self.linear3 = nn.Linear(128, 1, bias=False)
        if opt.weight_mode == 'dot':
            # self.activation = nn.Softplus()
            self.activation = nn.Sigmoid()
        elif opt.weight_mode == 'sample':
            self.activation = nn.Sigmoid()
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.xavier_uniform_(self.linear3.weight)
    
    def forward(self, user, pos_item, neg_item):
        state = self.generate_controller(user, pos_item, neg_item)
        a = self.linear1(state)
        b = torch.relu(self.linear2(a))
        c = torch.relu(self.linear3(b))
        res = self.activation(c)
        return res
    
    def generate_controller(self, user, pos, neg):
        # A_pdt = user * pos
        # a_pdt = user * neg
        # state = torch.cat((A_pdt, a_pdt), dim=1).squeeze()
        state = torch.cat((user, pos, neg), dim=1).squeeze()
        return state
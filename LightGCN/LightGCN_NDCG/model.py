import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
import utils as ut

class Model(nn.Module):
    def __init__(self, opt, Graph):
        super(Model, self).__init__()
        self.opt = opt
        self.A_split = opt.A_split
        self.dropout = opt.dropout
        self.n_layers = opt.n_layers
        self.num_users = opt.num_users
        self.num_items = opt.num_items
        self.user_embedding = nn.Embedding(opt.num_users, opt.user_dim)
        self.item_embedding = nn.Embedding(opt.num_items, opt.item_dim)
        self.user_embedding.weight.data.normal_(0, 1.0 / self.user_embedding.embedding_dim)
        self.item_embedding.weight.data.normal_(0, 1.0 / self.item_embedding.embedding_dim)
        self.Graph = Graph

    def propagate(self):
        """
        propagate methods for lightGCN
        """       
        users_emb = self.user_embedding.weight
        items_emb = self.item_embedding.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph        
        else:
            g_droped = self.Graph    
        
        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        #print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def forward(self, user_idx, item_idx):
        all_users, all_items = self.propagate()
        user = all_users[user_idx.long()]
        item = all_items[item_idx.long()]
        rating = (user * item).sum(dim=1).squeeze()
        return rating, user, item

    def predict(self, user_idx, all_users, all_items):
        user = all_users[user_idx.long()]
        rating = torch.matmul(user, all_items.t())
        return rating
    
    def predict_valid(self, user_idx, item_idx):
        all_users, all_items = self.propagate()
        user = all_users[user_idx.long()]
        item = all_items[item_idx.long()]
        rating = torch.matmul(user.unsqueeze(1), item.permute(0, 2, 1)).squeeze()
        return rating
    
    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g
    
    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph
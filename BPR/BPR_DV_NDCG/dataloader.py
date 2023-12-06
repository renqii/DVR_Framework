import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
last_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(last_dir)
import numpy as np
import torch
import scipy.sparse as sp
import argparse
from dataset import Beauty, Yelp2018, Gowalla, LastFM
import torch.utils.data as data
import torch.utils.data as dataloader

class DataLoader(object):
    def __init__(self, opt, dataset):
        self.opt = opt
        self.batch_size = opt.batch_size
        self.test_batch_size = opt.test_batch_size
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.train_dict = dataset.train_dict
        self.test_dict = dataset.test_dict
        self.num_train_interactions = np.array([len(item) for item in self.train_dict.values()]).sum()
        self.train_dataset = TrainDataset(self.train_dict, self.num_users, self.num_items)
        self.train_dataloader = data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_dataset = TestDataset(self.train_dict, self.test_dict, self.num_users, self.num_items)
        self.test_dataloader = data.DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False, num_workers=0)
        self.neg_val_items = torch.zeros((self.num_users, 100), dtype=torch.long)

    def valid_neg_sampling(self):
        neg_items = {}
        for user in range(self.num_users):
            neg_items[user] = []
            for idx in range(100):
                while True:
                    neg_item = torch.randint(0, self.num_items, (1,)).item()
                    if neg_item not in self.train_dict[user]:
                        break
                self.neg_val_items[user][idx] = neg_item

    def generate_valid(self, user_list, item_list):
        user_array = torch.concat(user_list)
        item_array = torch.concat(item_list)
        users = torch.zeros(user_array.shape[0], dtype=torch.long)
        items = torch.zeros((item_array.shape[0], 101), dtype=torch.long)
        for idx in range(user_array.shape[0]):
            uid = user_array[idx]
            users[idx] = uid
            items[idx][0] = item_array[idx]
            items[idx][1:] = self.neg_val_items[uid]
        return users, items

class ValidDataset(data.Dataset):
    def __init__(self, user_list, item_list, neg_val_items):
        self.neg_val_items = neg_val_items
        self.users, self.items = self.generate_valid(user_list, item_list)
        

    def generate_valid(self, user_list, item_list):
        user_array = torch.concat(user_list)
        item_array = torch.concat(item_list)
        users = torch.zeros(user_array.shape[0], dtype=torch.long)
        items = torch.zeros((item_array.shape[0], 101), dtype=torch.long)
        for idx in range(user_array.shape[0]):
            uid = user_array[idx]
            users[idx] = uid
            items[idx][0] = item_array[idx]
            items[idx][1:] = self.neg_val_items[uid]
        return users, items
    
    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        return self.users[idx], self.items[idx]

    
class TrainDataset(data.Dataset):
    def __init__(self, train_dict, num_users, num_items):
        self.train_dict = train_dict
        self.num_users = num_users
        self.num_items = num_items
        self.users, self.pos_items = self.generate_pair()
        self.neg_items = np.zeros(len(self.users)).astype(np.int32)

    def __len__(self):
        return len(self.users)
    
    def neg_sampling(self):
        for i in range(len(self.users)):
            user = self.users[i]
            while True:
                neg_item = np.random.randint(self.num_items)
                if neg_item not in self.train_dict[user]:
                    break
            self.neg_items[i] = neg_item
    
    def __getitem__(self, idx):
        uid = self.users[idx]
        pos_iid = self.pos_items[idx]
        neg_iid = self.neg_items[idx]
        return uid, pos_iid, neg_iid

    def generate_pair(self):
        users, pos_items = [], []
        for user, items in self.train_dict.items():
            for item in items:
                users.append(user)
                pos_items.append(item)
        return np.array(users).astype(np.int32), np.array(pos_items).astype(np.int32)


class TestDataset(data.Dataset):
    def __init__(self, train_dict, test_dict, num_users, num_items):
        self.train_dict = train_dict
        self.test_dict = test_dict
        self.num_users = num_users
        self.num_items = num_items
        self.test_keys = list(self.test_dict.keys())
    
    def __len__(self):
        return len(self.test_dict)
    
    def __getitem__(self, idx):
        key = self.test_keys[idx]
        test_items = self.test_dict[key]
        test_items_tensor = np.zeros(self.num_items)
        for item in test_items:
            test_items_tensor[item] = 1
        
        user_items = self.train_dict[key]
        user_items_tensor = torch.zeros(self.num_items)
        for item in user_items:
            user_items_tensor[item] = 1
        return key, test_items_tensor, user_items_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data arguments
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--test_batch_size', type=int, default=10)
    opt = parser.parse_args()
    dataset = Beauty()
    dataloader = DataLoader(opt, dataset)
    print(dataloader.num_interactions)
    # train_dataloader = dataloader.train_dataloader
    # test_dataloader = dataloader.test_dataloader
    # train_dataloader.dataset.neg_sampling()
    # for i, batch in enumerate(train_dataloader):
    #     batch_user, batch_item, batch_neg_item = batch
    # for i, batch in enumerate(test_dataloader):
    #     batch_user, batch_test_items, batch_user_items = batch

    
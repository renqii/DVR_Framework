import os
import pandas as pd
from tqdm import tqdm
import collections
import numpy as np
import pickle
import scipy.sparse as sp
data_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
class Beauty(object):
    def __init__(self):
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        train_path = os.path.join(data_dir, 'dataset/beauty/beauty_data.pkl')
        dataset = pickle.load(open(train_path, 'rb'))
        train_user_list = dataset['train_user_list'][1:]
        test_user_list = dataset['test_user_list'][1:]

        interaction_pair = []
        for user, item_set in enumerate(train_user_list):
            for item in list(item_set):
                interaction_pair.append([user, item, 1])
        for user, item_set in enumerate(test_user_list):
            for item in list(item_set):
                interaction_pair.append([user, item, 0])
        df = pd.DataFrame(interaction_pair, columns=['user', 'item', 'is_train'])
        df['user'] = df['user'].astype('category').cat.codes
        df['item'] = df['item'].astype('category').cat.codes
        num_users = len(df['user'].unique())
        num_items = len(df['item'].unique())

        train_df = df[df['is_train'] == 1]
        test_df = df[df['is_train'] == 0]

        train_csr = sp.csr_matrix((np.ones(train_df.shape[0]), (train_df['user'], train_df['item'])), shape=(num_users, num_items))
        test_csr = sp.csr_matrix((np.ones(test_df.shape[0]), (test_df['user'], test_df['item'])), shape=(num_users, num_items))
        
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for uid, item_list in enumerate(train_csr):
            for item in item_list.indices.tolist():
                train_dict[uid].append(item)
        
        for uid, item_list in enumerate(test_csr):
            for item in item_list.indices.tolist():
                test_dict[uid].append(item)
        return train_dict, test_dict, num_users, num_items

class CD(object):
    def __init__(self):
        self.data_path = os.path.join(data_dir, 'dataset/cd/amazon-cd.pkl')
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        with open(self.data_path, 'rb') as f:
            train_matrix, val_matrix, test_matrix, train_set, val_set, test_set = pickle.load(f)
        
        num_users = train_matrix.shape[0]
        num_items = train_matrix.shape[1]
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for user in range(num_users):
            train_dict[user] = list(train_set[user]) + list(val_set[user])
            test_dict[user] = list(test_set[user])
        return train_dict, test_dict, num_users, num_items


class Yelp2018(object):
    def __init__(self):
        self.data_path = os.path.join(data_dir, 'dataset/yelp2018/yelp2018.pkl')
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        with open(self.data_path, 'rb') as f:
            train_matrix, val_matrix, test_matrix, train_set, val_set, test_set = pickle.load(f)
        
        num_users = train_matrix.shape[0]
        num_items = train_matrix.shape[1]
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for user in range(num_users):
            train_dict[user] = list(train_set[user]) + list(val_set[user])
            test_dict[user] = list(test_set[user])
        return train_dict, test_dict, num_users, num_items


class Gowalla(object):
    def __init__(self):
        self.dir_path = os.path.join(data_dir, 'dataset/gowalla')
        self.train_file = 'train.txt'
        self.test_file = 'test.txt'
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        train_path = os.path.join(self.dir_path, self.train_file)
        test_path = os.path.join(self.dir_path, self.test_file)

        max_uid, max_iid = 0, 0
        train_dict = collections.defaultdict(list)
        with open(train_path, 'r') as f:
            train_data = f.readlines()
            for line in train_data:
                line = line.strip().split(' ')
                uid = int(line[0])
                iid_list = [int(iid) for iid in line[1:]]
                train_dict[uid] = iid_list
                max_uid = max(max_uid, uid)
                max_iid = max(max_iid, max(iid_list))
        
        test_dict = collections.defaultdict(list)
        with open(test_path, 'r') as f:
            test_data = f.readlines()
            for line in test_data:
                line = line.strip().split(' ')
                uid = int(line[0])
                iid_list = [int(iid) for iid in line[1:]]
                test_dict[uid] = iid_list
                max_uid = max(max_uid, uid)
                max_iid = max(max_iid, max(iid_list))
        
        num_users = max_uid + 1
        num_items = max_iid + 1
        return train_dict, test_dict, num_users, num_items

class LastFM(object):
    def __init__(self):
        self.data_path = os.path.join(data_dir, 'dataset/lastfm/lastfm.pkl')
        self.train_dict, self.test_dict, self.num_users, self.num_items = self.generate_dataset()

    def generate_dataset(self):
        with open(self.data_path, 'rb') as f:
            train_matrix, val_matrix, test_matrix, train_set, val_set, test_set = pickle.load(f)
        
        num_users = train_matrix.shape[0]
        num_items = train_matrix.shape[1]
        train_dict = collections.defaultdict(list)
        test_dict = collections.defaultdict(list)
        for user in range(num_users):
            train_dict[user] = list(train_set[user]) + list(val_set[user])
            test_dict[user] = list(test_set[user])
        return train_dict, test_dict, num_users, num_items

if __name__ == '__main__':
    dataset1 = Beauty()
    dataset2 = Yelp2018()
    dataset3 = Gowalla()
    dataset4 = LastFM()
    dataset5 = CD()
    print(dataset1.num_users)
    print(dataset1.num_items)
    print(dataset2.num_users)
    print(dataset2.num_items)
    print(dataset3.num_users)
    print(dataset3.num_items)
    print(dataset4.num_users)
    print(dataset4.num_items)
    print(dataset5.num_users)
    print(dataset5.num_items)
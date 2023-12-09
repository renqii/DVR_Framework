import torch
import random
import numpy as np
import os
from torch.autograd import Variable
import logging
import torch.nn as nn
import utils as ut

def set_device(cuda_id):
    device = torch.device("cuda:{}".format(str(cuda_id)) if(torch.cuda.is_available()) else "cpu")
    # device = torch.device("cpu")
    return device

def init_seed(opt, seed=3333):
    if opt.cuda:
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def BPR_loss(pos_scores, neg_scores):
    # loss = torch.mean(softplus(neg_scores - pos_scores))
    # loss = - torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
    loss = - torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
    return loss

def BCE_loss(pos_scores, neg_scores):
    pos_labels, neg_labels = torch.ones(pos_scores.size()), torch.zeros(neg_scores.size())
    loss = nn.BCELoss()(pos_scores, pos_labels)
    loss += nn.BCELoss()(neg_scores, neg_labels)
    return loss

def change_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

class Logger(object):
    def __init__(self, file_name='my.log'):
        self.logger = logging.getLogger()  # 不加名称设置root logger
        self.logger.setLevel(logging.DEBUG)
        os.environ['NUMEXPR_MAX_THREADS'] = '16'
        LOG_FORMAT = "%(asctime)s - %(message)s"
        DATE_FORMAT = "%m-%d %H:%M"
        # logging.basicConfig(filename='my.log', level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

        formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

        # 使用FileHandler输出到文件
        base_path = '/home/s2mc/Programs/CausalRec/log/'
        path = base_path + file_name
        self.fh = logging.FileHandler(path)
        self.fh.setLevel(logging.DEBUG)
        self.fh.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)
        self.ch.setFormatter(formatter)

        # 添加两个Handler
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def pprint(self, message):
        self.logger.info(message)
        # self.logger.debug(message)

def set_logger(file_name):
    logger = Logger(file_name)
    return logger

class SlidingWindow(object):
    def __init__(self, opt):
        self.window_type = opt.window_type
        self.window_size = opt.window_size
        self.window = []
    
    def sliding(self, data):
        if len(self.window) > self.window_size:
            self.window.pop(0)
        # res = np.array(self.window).mean()
        if len(self.window) == 0:
            res = 0.0
        else:
            if self.window_type == 'mean':
                res = np.array(self.window).mean()
            elif self.window_type == 'max':
                res = np.array(self.window).max()
        self.window.append(data)
        return res
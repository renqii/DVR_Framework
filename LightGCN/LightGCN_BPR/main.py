import sys
import os
cur_dir = os.path.dirname(os.path.abspath(__file__))
last_dir = os.path.dirname(cur_dir)
sys.path.append(cur_dir)
sys.path.append(last_dir)
import torch
import argparse
import logging
import datetime
import numpy as np
import json
from time import time
import pickle
from torch.distributions import binomial
from torch.utils.tensorboard import SummaryWriter
from dataloader import DataLoader
from model import Model
from value import Value
import utils as ut
from eval_metrics import *
from dataset import Beauty, CD, Yelp2018, Gowalla, LastFM

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser()
# data arguments
parser.add_argument('-c', '--cuda', type=int, required=True)
parser.add_argument('-d', '--dataset', type=str, choices=['beauty', 'cd', 'yelp2018', 'lastfm', 'gowalla'], required=True)

parser.add_argument('--num_users', type=int, default=0)
parser.add_argument('--num_items', type=int, default=0)
parser.add_argument('--user_dim', type=int, default=64)
parser.add_argument('--item_dim', type=int, default=64)
parser.add_argument('--valid_interval', type=int, default=10)
parser.add_argument('--save_model_interval', type=int, default=10)
known_args, _ = parser.parse_known_args()
hyper_params = json.load(open(os.path.join(cur_dir, 'config.json')))[known_args.dataset]
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--bpr_lr', type=float, default=0.001)
parser.add_argument('--dve_lr', type=float, default=0.001)
parser.add_argument('--bpr_weight_decay', type=float, default=0.001)
parser.add_argument('--dve_weight_decay', type=float, default=0.001)
parser.add_argument('--l2', type=int, default=0.0001)
parser.add_argument('--dropout', type=bool, default=False)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--A_fold', type=int, default=100)
parser.add_argument('--A_split', type=bool, default=False)
parser.add_argument('--dve_update_interval', type=int, default=1)
parser.add_argument('--dve_print_interval', type=int, default=100)
parser.add_argument('--mode', type=str, choices=['sample', 'dot'], default='sample')
parser.add_argument('--window_size', type=int, default=100)
parser.add_argument('--window_type', type=str, choices=['mean', 'max'], default='mean')
parser.add_argument('--reward', type=str, choices=['bpr'], default='bpr')
parser.add_argument('--value', type=str, choices=['triplet', 'overall'], default='overall')
parser.add_argument('--weighted', type=bool, default=False)
parser.add_argument('-lm', '--load_model', type=int, default=0)
parser.add_argument('-sm', '--save_model', type=bool, default=False)
parser.add_argument('--load_path', type=str, default=None)


parser.set_defaults(**hyper_params)

opt = parser.parse_args()

ut.init_seed(opt)

device = ut.set_device(opt.cuda)
opt.device = device

opt.log_name = ''.join([datetime.datetime.now().strftime("%d-%m-%Y"), \
                        '_Bsz_', str(opt.batch_size), '_bpr_lr_', str(opt.bpr_lr), '_dve_lr_', str(opt.dve_lr), \
                            '_BprWeightDecay_', str(opt.bpr_weight_decay), '_dve_weight_decay_', str(opt.dve_weight_decay), \
                                '_DveUpdateInterval_', str(opt.dve_update_interval), \
                                    '_LoadModel_', str(opt.load_model), \
                                        '_Mode_', str(opt.mode), \
                                            '_WindowSize_', str(opt.window_size), \
                                                '_WindowType_', str(opt.window_type), \
                                                    '_Reward_', str(opt.reward), \
                                                        '_Value_', str(opt.value), \
                                                            '_Weighted_', str(opt.weighted)])

if not os.path.exists(os.path.join(cur_dir, 'log')):
    os.mkdir(os.path.join(cur_dir, 'log'))
if not os.path.exists(os.path.join(cur_dir, 'log', opt.dataset)):
    os.mkdir(os.path.join(cur_dir, 'log', opt.dataset))
if not os.path.exists(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name)):
    os.mkdir(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name))
if not os.path.exists(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint')):
    os.mkdir(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint'))
save_model_path = os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint')

writer = SummaryWriter(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name), flush_secs=10)
handler = logging.FileHandler(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, opt.log_name + '.log'), mode='w')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

save_model_path = os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint')

pid = os.getpid()
logger.info('pid: {}'.format(pid))

logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

logger.info(opt)

logger.info(hyper_params)

t0 = time()

if opt.dataset == 'beauty':
    dataset = Beauty(opt)
elif opt.dataset == 'cd':
    dataset = CD(opt)
elif opt.dataset == 'yelp2018':
    dataset = Yelp2018(opt)
elif opt.dataset == 'lastfm':
    dataset = LastFM(opt)
elif opt.dataset == 'gowalla':
    dataset = Gowalla(opt)


dataloader = DataLoader(opt, dataset)
train_dataloader = dataloader.train_dataloader
test_dataloader = dataloader.test_dataloader

opt.num_users = dataset.num_users
opt.num_items = dataset.num_items

model = Model(opt, dataset.Graph.to(device)).to(device)
# load model
if opt.load_model:
    logger.info('load model from {}'.format(opt.load_path))
    load_dict = torch.load(opt.load_path, map_location=device)
    model.load_state_dict(load_dict)
    # model.load_state_dict(load_dict['state_dict'])
    # start_epoch = load_dict['epoch'] + 1
    start_epoch = 0
else:
    start_epoch = 0

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.bpr_lr, weight_decay=opt.bpr_weight_decay)

data_estimator = Value(opt).to(device)
dve_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, data_estimator.parameters()), lr=opt.dve_lr, weight_decay=opt.dve_weight_decay)

sliding_window = ut.SlidingWindow(opt)

best_valid_eval = [0.0, 0.0, 0.0, 0.0]
dve_reward = []
dve_sel_prob = []
dve_data_value = []
for epoch_num in range(start_epoch, opt.epoch):
    t1 = time()
    model.train()
    epoch_loss = 0.0
    loss_log = {'bpr_loss_weighted': 0.0, 'bpr_loss_ori': 0.0, 'reward':0.0}
    recall_10_list = [0.0]
    train_dataloader.dataset.neg_sampling()
    for i, batch in enumerate(train_dataloader):
        model.train()
        # compute bpr loss
        batch_user, batch_item, batch_neg_item = batch
        batch_user, batch_item, batch_neg_item = batch_user.to(device), batch_item.to(device), batch_neg_item.to(device)
        # targets_prediction, batch_user_embedding, batch_item_embedding = model(batch_user, batch_item)
        # negatives_prediction, _, batch_neg_item_embedding = model(batch_user, batch_neg_item)
        targets_prediction, _, _ = model(batch_user, batch_item)
        negatives_prediction, _, _ = model(batch_user, batch_neg_item)
        # compute the BPR loss
        bpr_loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)

        batch_user_embedding = model.user_embedding(batch_user).detach().clone()
        batch_item_embedding = model.item_embedding(batch_item).detach().clone()
        batch_neg_item_embedding = model.item_embedding(batch_neg_item).detach().clone()
        # data estimator
        data_value = data_estimator(batch_user_embedding.detach().clone(), batch_item_embedding.detach().clone(), batch_neg_item_embedding.detach().clone())
        data_value = data_value.squeeze()
        dve_data_value.append(data_value)

        # compute weighted bpr
        if opt.mode == 'sample':
            distribution = binomial.Binomial(total_count=1, probs=data_value)
            sel_prob_curr = distribution.sample()
            dve_sel_prob.append(sel_prob_curr)
            bpr_loss_weighted = torch.sum(bpr_loss * sel_prob_curr)
        elif opt.mode == 'dot':
            bpr_loss_weighted = torch.sum(bpr_loss * data_value.detach().clone())

        # optimize bpr model
        # compute the l2 regularization
        reg_los = (1/2)*(torch.norm(batch_user_embedding) ** 2 + torch.norm(batch_item_embedding) ** 2 + torch.norm(batch_neg_item_embedding) ** 2) / float(len(batch_user))
        loss = bpr_loss_weighted + opt.l2 * reg_los
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # compute reward
        bpr_loss_ori = torch.sum(bpr_loss).item()
        if opt.window_size == 0:
            reward = - bpr_loss_ori
        else:
            reward_mean = sliding_window.sliding(-bpr_loss_ori)
            reward = - bpr_loss_ori - reward_mean
        dve_reward.append(reward)
        
        if (epoch_num * len(train_dataloader)  + i) % opt.dve_update_interval == 0:
            data_estimator.train()
            reward = np.array(dve_reward).mean()
            data_value = torch.concat(dve_data_value, dim=0)
            if opt.mode == 'sample':
                sel_prob = torch.concat(dve_sel_prob, dim=0)
                prob = torch.sum(sel_prob * torch.log(data_value + 1e-8) + \
                        (1 - sel_prob) * torch.log(1 - data_value + 1e-8))
            elif opt.mode == 'dot':
                prob = torch.sum(torch.log(data_value + 1e-8))
            dve_loss = reward * prob
            dve_optimizer.zero_grad()
            dve_loss.backward()
            dve_optimizer.step()
            dve_reward = []
            dve_data_value = []
            dve_sel_prob = []

        epoch_loss += bpr_loss_weighted.item()
        loss_log['bpr_loss_weighted'] += bpr_loss_weighted.item()
        loss_log['bpr_loss_ori'] += bpr_loss_ori
        loss_log['reward'] += reward

        if (epoch_num * len(train_dataloader)  + i) % opt.dve_print_interval == 0:
            format_str = 'Epoch {:3d} iter {:3d} max_weight {:.4f} min_weight {:.4f} mean_weight {:.4f}'
            logger.info(format_str.format(epoch_num, i, torch.max(data_value).item(), torch.min(data_value).item(), torch.mean(data_value).item()))

    epoch_loss /= len(train_dataloader)
    for key in loss_log.keys():
        loss_log[key] /= len(train_dataloader)
    
    bpr_pre = epoch_loss

    t2 = time()
    format_str = 'Epoch {:3d} [{:.1f} s]  loss={:.4f}: bpr_loss_weighted={:.4f}, bpr_loss_ori={:.4f}, reward={:.4f}'
    logger.info(format_str.format(epoch_num, t2 - t1, epoch_loss, loss_log['bpr_loss_weighted'], loss_log['bpr_loss_ori'], loss_log['reward']))
    writer.add_scalar('train/loss', epoch_loss, epoch_num)
    writer.add_scalar('train/bpr_loss_weighted', loss_log['bpr_loss_weighted'], epoch_num)
    writer.add_scalar('train/bpr_loss_ori', loss_log['bpr_loss_ori'], epoch_num)
    writer.add_scalar('train/reward', loss_log['reward'], epoch_num)

# test
    if epoch_num > 0 and epoch_num % opt.valid_interval == 0:
        model.eval()
        all_users, all_items = model.propagate()
        for idx, batch_test_data in enumerate(test_dataloader):
            batch_user_ids, batch_test_items, batch_user_items = batch_test_data
            batch_user_ids = batch_user_ids.to(device)
            rating_pred = model.predict(batch_user_ids, all_users, all_items)
            rating_pred = rating_pred.reshape(-1, rating_pred.shape[-1])
            rating_pred = rating_pred.cpu().data.numpy().copy()
            rating_pred[batch_user_items > 0] = -np.inf
            if idx == 0:
                pred_list = rating_pred
                test_matrix = batch_test_items
            else:
                pred_list = np.append(pred_list, rating_pred, axis=0)
                test_matrix = np.append(test_matrix, batch_test_items, axis=0)
        r_10 = Recall_at_k_batch(pred_list, test_matrix, 10).mean()
        n_10 = NDCG_binary_at_k_batch(pred_list, test_matrix, 10).mean()
        r_20 = Recall_at_k_batch(pred_list, test_matrix, 20).mean()
        n_20 = NDCG_binary_at_k_batch(pred_list, test_matrix, 20).mean()

        metrics = []
        metrics.append("Recall@10 {:.5f}".format(r_10))
        metrics.append("NDCG@10 {:.5f}".format(n_10))
        metrics.append("Recall@20 {:.5f}".format(r_20))
        metrics.append("NDCG@20 {:.5f}".format(n_20))
        writer.add_scalar('valid/Recall@10', r_10, epoch_num)
        writer.add_scalar('valid/NDCG@10', n_10, epoch_num)
        writer.add_scalar('valid/Recall@20', r_20, epoch_num)
        writer.add_scalar('valid/NDCG@20', n_20, epoch_num)

        logger.info('\nvalid epoch {} '.format(epoch_num) + " ".join(metrics))
        logger.info("Evaluation time:{}".format(time() - t2))
        

        if r_20 > best_valid_eval[2]:
            best_valid_eval = [r_10, n_10, r_20, n_20]
            if opt.save_model:
                save_obj = {'state_dict': model.state_dict(), 'epoch': epoch_num, 'best_valid_eval': best_valid_eval}
                torch.save(model.state_dict(), os.path.join(save_model_path, 'model_' + str(epoch_num) + '.pt'))

metrics = []
metrics.append("Recall@10 {:.5f}".format(best_valid_eval[0]))
metrics.append("NDCG@10 {:.5f}".format(best_valid_eval[1]))
metrics.append("Recall@20 {:.5f}".format(best_valid_eval[2]))
metrics.append("NDCG@20 {:.5f}".format(best_valid_eval[3]))
logger.info('\nBest valid' + " ".join(metrics))

logger.info("Time cost: {:.2f}".format((time() - t0) / 3600))
logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
logger.info("\n")
logger.info("\n")

# save model
torch.save(model.state_dict(), os.path.join(save_model_path, 'model_' + str(opt.epoch) + '.pt'))
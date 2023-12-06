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
from dataloader import DataLoader, ValidDataset
from bpr import BPR
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
parser.add_argument('--num_interactions', type=int, default=0)
parser.add_argument('--user_dim', type=int, default=64)
parser.add_argument('--item_dim', type=int, default=64)
parser.add_argument('--valid_interval', type=int, default=10)
parser.add_argument('--save_model_interval', type=int, default=10)
known_args, _ = parser.parse_known_args()
hyper_params = json.load(open(os.path.join(cur_dir, 'config.json')))[known_args.dataset]
parser.add_argument('-lm', '--load_model', type=int, default=0)
parser.add_argument('-sm', '--save_model', type=bool, default=False)
parser.add_argument('--load_path', type=str, default=None)
parser.add_argument('--epoch', type=int, default=2000)
parser.add_argument('--batch_size', type=int, default=5000)
parser.add_argument('--test_batch_size', type=int, default=1000)

parser.add_argument('--bpr_lr', type=float, default=0.005)
parser.add_argument('--dve_lr', type=float, default=0.001)
parser.add_argument('--bpr_weight_decay', type=float, default=0.001)
parser.add_argument('--dve_weight_decay', type=float, default=0.001)
parser.add_argument('--dve_update_interval', type=int, default=10)
parser.add_argument('--dve_print_interval', type=int, default=10)
parser.add_argument('--weight_mode', type=str, choices=['sample', 'dot'], default='sample')
parser.add_argument('--window_size', type=int, default=0)
parser.add_argument('--reward', type=str, choices=['recall', 'ndcg'], default='ndcg')
parser.set_defaults(**hyper_params)
opt = parser.parse_args()

ut.init_seed(opt)

device = ut.set_device(opt.cuda)

opt.log_name = ''.join([datetime.datetime.now().strftime("%d-%m-%Y"), \
                        '_bsz_', str(opt.batch_size), '_bpr_lr_', str(opt.bpr_lr), '_dve_lr_', str(opt.dve_lr), \
                        '_bpr_weight_decay_', str(opt.bpr_weight_decay), '_dve_weight_decay_', str(opt.dve_weight_decay), \
                            '_dve_update_interval_', str(opt.dve_update_interval), \
                                '_load_model_', str(opt.load_model), \
                                    '_weight_mode_', str(opt.weight_mode), \
                                        '_window_size_', str(opt.window_size)])

if not os.path.exists(os.path.join(cur_dir, 'log')):
    os.mkdir(os.path.join(cur_dir, 'log'))
if not os.path.exists(os.path.join(cur_dir, 'log', opt.dataset)):
    os.mkdir(os.path.join(cur_dir, 'log', opt.dataset))
if not os.path.exists(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name)):
    os.mkdir(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name))
if not os.path.exists(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint')):
    os.mkdir(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint'))
save_model_path = os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint')

writer = SummaryWriter(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name))
handler = logging.FileHandler(os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, opt.log_name + '.log'), mode='w')
handler.setLevel(logging.INFO)
logger.addHandler(handler)

save_model_path = os.path.join(cur_dir, 'log', opt.dataset, opt.log_name, 'checkpoint')

pid = os.getpid()
logger.info('pid: {}'.format(pid))

logger.info(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

logger.info(opt)

logger.info(hyper_params)
writer.add_text('hyper_params', str(hyper_params))

t0 = time()

if opt.dataset == 'beauty':
    dataset = Beauty()
elif opt.dataset == 'cd':
    dataset = CD()
elif opt.dataset == 'yelp2018':
    dataset = Yelp2018()
elif opt.dataset == 'lastfm':
    dataset = LastFM()
elif opt.dataset == 'gowalla':
    dataset = Gowalla()

dataloader = DataLoader(opt, dataset)
train_dataloader = dataloader.train_dataloader
test_dataloader = dataloader.test_dataloader

opt.num_users = dataset.num_users
opt.num_items = dataset.num_items

model = BPR(opt).to(device)
# load model
if opt.load_model:
    logger.info('load model from {}'.format(opt.load_path))
    load_dict = torch.load(opt.load_path)
    model.load_state_dict(load_dict)
    # model.load_state_dict(load_dict['state_dict'])
    # start_epoch = load_dict['epoch'] + 1
    start_epoch = 0
else:
    start_epoch = 0

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.bpr_lr, weight_decay=opt.bpr_weight_decay)

data_estimator = Value(opt).to(device)
dve_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, data_estimator.parameters()), lr=opt.dve_lr, weight_decay=opt.dve_weight_decay)

sliding_window = ut.SlidingWindow(opt.window_size)

best_valid_eval = [0.0, 0.0, 0.0, 0.0]
dve_reward = []
dve_sel_prob = []
dve_data_value = []
dve_iter_count = 0
valid_user_list = []
valid_item_list = []
sel_prob_list = []
data_value_list = []
for epoch_num in range(start_epoch, opt.epoch):
    t1 = time()
    loss_log = {'bpr_loss_weighted': 0.0, 'bpr_loss_ori': 0.0, 'reward':0.0}
    recall_list = [0.0]
    train_dataloader.dataset.neg_sampling()
    dataloader.valid_neg_sampling()

    for i, batch in enumerate(train_dataloader):
        model.train()
        batch_user, batch_item, batch_neg_item = batch
        valid_user_list.append(batch_user)
        valid_item_list.append(batch_item)
        batch_user, batch_item, batch_neg_item = batch_user.to(device), batch_item.to(device), batch_neg_item.to(device)
        targets_prediction = model(batch_user, batch_item)
        negatives_prediction = model(batch_user, batch_neg_item)
        bpr_loss = -torch.log(torch.sigmoid(targets_prediction - negatives_prediction) + 1e-8)
        bpr_loss_ori = torch.sum(bpr_loss)
        
        # data estimator
        batch_user_embedding = model.user_embedding(batch_user).detach().clone()
        batch_item_embedding = model.item_embedding(batch_item).detach().clone()
        batch_neg_item_embedding = model.item_embedding(batch_neg_item).detach().clone()
        data_value = data_estimator(batch_user_embedding, batch_item_embedding, batch_neg_item_embedding)
        data_value = data_value.squeeze()
        # compute weighted bpr
        if opt.weight_mode == 'sample':
            distribution = binomial.Binomial(total_count=1, probs=data_value)
            sel_prob = distribution.sample()
            dve_sel_prob.append(sel_prob)
            bpr_loss_weighted = torch.sum(bpr_loss * sel_prob)
            sel_prob_list.append(sel_prob)
        elif opt.weight_mode == 'dot':
            bpr_loss_weighted = torch.sum(bpr_loss * data_value.detach().clone())

        optimizer.zero_grad()
        bpr_loss_weighted.backward()
        optimizer.step()

        
        data_value_list.append(data_value)
        loss_log['bpr_loss_weighted'] += bpr_loss_weighted.item()
        loss_log['bpr_loss_ori'] += bpr_loss_ori.item()

        if (epoch_num * len(train_dataloader)  + i) % opt.dve_print_interval == 0:
            format_str = 'Epoch {:3d} iter {:3d} max_weight {:.4f} min_weight {:.4f} mean_weight {:.4f}'
            logger.info(format_str.format(epoch_num, i, torch.max(data_value).item(), torch.min(data_value).item(), torch.mean(data_value).item()))

        if (epoch_num * len(train_dataloader)  + i) % opt.dve_update_interval == 0:
            dve_iter_count += 1
            writer.add_scalar('max_weight', torch.max(data_value).item(), dve_iter_count)
            writer.add_scalar('min_weight', torch.min(data_value).item(), dve_iter_count)
            writer.add_scalar('mean_weight', torch.mean(data_value).item(), dve_iter_count)

            # validation
            t3 = time()
            model.eval()
            valid_dataset = ValidDataset(valid_user_list, valid_item_list, dataloader.neg_val_items)
            valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=5000, shuffle=False)
            valid_user_list, valid_item_list = [], []
            recall_list = []
            ndcg_list = []
            for idx, batch in enumerate(valid_dataloader):
                valid_user, valid_item = batch
                valid_user, valid_item = valid_user.to(device), valid_item.to(device)
                rating_pred = model.predict_valid(valid_user, valid_item)
                rating_pred = rating_pred.cpu().data
                sorted_indices = torch.argsort(rating_pred, dim=1, descending=True)
                topk_indices = sorted_indices[:, :1]
                recall = torch.sum(topk_indices == 0) / topk_indices.shape[0]
                ndcg = (1 / torch.log2(topk_indices + 2)).mean()
                recall_list.append(recall.item())
                ndcg_list.append(ndcg.item())
            recall_val = np.array(recall_list).mean()
            ndcg_val = np.array(ndcg_list).mean()
            
            data_estimator.train()
            
            data_value_iter = torch.cat(data_value_list, dim=0)
            data_value_list = []
            
            if opt.weight_mode == 'sample':
                sel_prob_iter = torch.cat(sel_prob_list, dim=0)
                sel_prob_list = []
                prob = torch.sum(sel_prob_iter * torch.log(data_value_iter + 1e-8) + \
                        (1 - sel_prob_iter) * torch.log(1 - data_value_iter + 1e-8))
            elif opt.weight_mode == 'dot':
                prob = torch.sum(torch.log(data_value_iter + 1e-8))
            
            if opt.reward == 'ndcg':
                reward = ndcg_val
            elif opt.reward == 'recall':
                reward = recall_val
                
            if opt.window_size is not 0:
                reward_mean = sliding_window.sliding(reward)
                reward = reward - reward_mean

            dve_reward.append(reward)

            policy_grad = reward * prob
            dve_loss = policy_grad

            dve_optimizer.zero_grad()
            dve_loss.backward()
            dve_optimizer.step()
            format_str = 'Epoch {:3d} iter {:3d} [{:.1f} s]: dve_loss {:.4f} policy_grad {:.4f} recall {:.4f} ndcg {:.4f} reward {:.4f} prob {:4f}'
            logger.info(format_str.format(epoch_num, i, time() - t3, dve_loss.item(), policy_grad.item(), recall_val.item(), ndcg_val.item(), reward.item(), prob.item()))
            writer.add_scalar('dve_loss', dve_loss.item(), dve_iter_count)
            writer.add_scalar('policy_grad', policy_grad.item(), dve_iter_count)
            writer.add_scalar('recall', recall_val.item(), dve_iter_count)
            writer.add_scalar('ndcg', ndcg_val.item(), dve_iter_count)
            writer.add_scalar('reward', reward.item(), dve_iter_count)
            writer.add_scalar('prob', prob.item(), dve_iter_count)


    for key in loss_log.keys():
        loss_log[key] /= len(train_dataloader)

    t2 = time()
    format_str = 'Epoch {:3d} [{:.1f} s]: bpr_loss_weighted={:.4f}, bpr_loss_ori={:.4f}'
    logger.info(format_str.format(epoch_num, t2 - t1, loss_log['bpr_loss_weighted'], loss_log['bpr_loss_ori']))

    writer.add_scalar('train/loss', loss_log['bpr_loss_ori'], epoch_num)
    writer.add_scalar('train/bpr_loss_weighted', loss_log['bpr_loss_weighted'], epoch_num)

# test
    if epoch_num > 0 and epoch_num % opt.valid_interval == 0:
        model.eval()
        for idx, batch_test_data in enumerate(test_dataloader):
            batch_user_ids, batch_test_items, batch_user_items = batch_test_data
            batch_user_ids = batch_user_ids.to(device)
            rating_pred = model.predict(batch_user_ids)
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
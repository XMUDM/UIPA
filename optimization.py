import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
from torch import nn, optim
import sys
import time
import random
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from util import *
import math

def SampleNegItem(user, m_items, positems):
    while True:
        negitem = np.random.randint(0, m_items)
        if negitem in positems:
            continue
        else:
            break
    return negitem

class CandidateDataset(Dataset):
    def __init__(self, u, v_p, v_n):
        self.len = u.shape[0]
        self.u_data = torch.as_tensor(u,dtype=torch.long)
        self.v_p_data = torch.as_tensor(v_p,dtype=torch.long)
        self.v_n_data = torch.as_tensor(v_n,dtype=torch.long)

    def __getitem__(self, index):
        return self.u_data[index], self.v_p_data[index], self.v_n_data[index]

    def __len__(self):
        return self.len

def UniformSample_original(dict_interactions, tr_u, n_users, m_items):
    """
    the original impliment of BPR Sampling in LightGCN
    :return:
        np.array
    """
    user_num = len(tr_u)
    users = np.random.randint(0, n_users, user_num)
    S = []
    for i, user in enumerate(users):
        posForUser = dict_interactions[user][:-(math.ceil(len(dict_interactions[user]) * 0.1)*2)]
        if len(posForUser) == 0:
            continue
        posindex = np.random.randint(0, len(posForUser))
        positem = posForUser[posindex]
        while True:
            negitem = np.random.randint(0, m_items)
            if negitem in posForUser:
                continue
            else:
                break
        S.append([user, positem, negitem])
    return np.array(S)

def shuffle(*arrays, **kwargs):
    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def Train_on_epoch(Recmodel, config, dictBank):
    prototype_loss_weight = config['prototype_loss_weight']
    aver_loss, aver_pre_loss, aver_reg_loss, aver_prototype_loss = 0., 0., 0., 0.
    total_batch = 0

    lr = config['lr']
    opt = optim.Adam(Recmodel.parameters(), lr=lr)
    Recmodel.train()

    # train source
    m_items = dictBank.m_items_s
    n_users = dictBank.n_users_s
    tr_u = dictBank.train_users_s
    dict_interactions = dictBank.dict_interactions_s
    is_target = 0

    S = UniformSample_original(dict_interactions, tr_u, n_users, m_items)

    users = torch.Tensor(S[:, 0]).long()
    all_users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()
    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch += len(users) // config['bpr_batch_size'] + 1
    if Recmodel.init_c == 1 and Recmodel.init_pseudo_s == 0:
        Recmodel.init_pseudo_info(all_users.cuda(), posItems.cuda(), negItems.cuda(), is_target=0)
        Recmodel.init_pseudo_s = 1
    for (batch_i, (u, v_p, v_n)) in enumerate(minibatch(users,
                                            posItems,
                                            negItems,
                                            batch_size=config['bpr_batch_size'])):
        Recmodel.batch_i = batch_i
        pre_loss, reg_loss, prototype_loss = Recmodel.getLoss(u.cuda(), v_p.cuda(), v_n.cuda(), is_target)

        reg_loss = reg_loss * 0.0005 # reg loss weight
        prototype_loss = prototype_loss * prototype_loss_weight
        loss = pre_loss + reg_loss + prototype_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        aver_loss += loss.cpu().item()
        aver_pre_loss += pre_loss.cpu().item()
        aver_reg_loss += reg_loss.cpu().item()
        aver_prototype_loss += prototype_loss.cpu().item()

    # train target
    m_items = dictBank.m_items_t
    n_users = dictBank.n_users_t
    tr_u = dictBank.train_users_t
    dict_interactions = dictBank.dict_interactions_t
    is_target = 1

    S = UniformSample_original(dict_interactions, tr_u, n_users, m_items)

    users = torch.Tensor(S[:, 0]).long()
    all_users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users, posItems, negItems = shuffle(users, posItems, negItems)
    total_batch += len(users) // config['bpr_batch_size'] + 1
    if Recmodel.init_c == 1 and Recmodel.init_pseudo_t == 0:
        Recmodel.init_pseudo_info(all_users.cuda(), posItems.cuda(), negItems.cuda(), is_target=1)
        Recmodel.init_pseudo_t = 1
    for (batch_i, (u, v_p, v_n)) in enumerate(minibatch(users,
                                             posItems,
                                             negItems,
                                             batch_size=config['bpr_batch_size'])):

        Recmodel.batch_i = batch_i
        pre_loss, reg_loss, prototype_loss = Recmodel.getLoss(u.cuda(), v_p.cuda(), v_n.cuda(), is_target)

        reg_loss = reg_loss * 0.0005 #reg loss weight
        prototype_loss = prototype_loss * prototype_loss_weight
        loss = pre_loss + reg_loss + prototype_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        aver_loss += loss.cpu().item()
        aver_pre_loss += pre_loss.cpu().item()
        aver_reg_loss += reg_loss.cpu().item()
        aver_prototype_loss += prototype_loss.cpu().item()

    aver_loss = aver_loss / total_batch
    aver_pre_loss = aver_pre_loss / total_batch
    aver_reg_loss = aver_reg_loss / total_batch
    aver_prototype_loss = aver_prototype_loss / total_batch
    
    return aver_loss, aver_pre_loss, aver_reg_loss, aver_prototype_loss


def RecallPrecision_ATk(test_data, r, k):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred / recall_n)
    precis = np.sum(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}

def HRatK(groundtruth, r, k):
    right_pred = r[:, :k].sum(1)
    interactions = np.array([len(groundtruth[i]) for i in range(len(groundtruth))])
    hitrate = np.sum(right_pred) / np.sum(interactions)
    return hitrate

def NDCGatK(test_data,r,k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r).astype('float')

def test_one_batch(X,config):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = getLabel(groundTrue, sorted_items)
    hr, pre, recall, ndcg = [], [], [], []
    for k in config["topks"]:
        ret = RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        hr.append(HRatK(groundTrue, r, k))
        ndcg.append(NDCGatK(groundTrue, r, k))
    return {'hr':np.array(hr),
            'recall': np.array(recall),
            'precision': np.array(pre),
            'ndcg': np.array(ndcg)}

def getUserPosItems(users, dict_interactions):
    posItems = []
    for user in users:
        posItems.append(dict_interactions[user][:-(math.ceil(len(dict_interactions[user]) * 0.1)*2)])
    return posItems

def Test(Recmodel, config, dict_interactions, is_val, is_target):
    Recmodel.eval()
    max_K = max(config["topks"])
    multicore = config['multicore']
    if multicore == 1:
        pool = multiprocessing.Pool(12)
    results = {'hr': np.zeros(len(config["topks"])),
               'ndcg': np.zeros(len(config["topks"])),
               'recall': np.zeros(len(config["topks"])),
               'precision': np.zeros(len(config["topks"]))
               }
    
    with torch.no_grad():
        users = list(dict_interactions.keys())
        users_list = []
        rating_list = []
        groundTrue_list = []

        allPos = getUserPosItems(users, dict_interactions)
        if is_val:
            groundTrue = [dict_interactions[u][-(math.ceil(len(dict_interactions[u]) * 0.1)*2) : -math.ceil(len(dict_interactions[u]) * 0.1)] for u in users]
        else:
            groundTrue = [dict_interactions[u][-math.ceil(len(dict_interactions[u]) * 0.1) : ] for u in users]
        users = torch.Tensor(users).long()
        if is_target:
            rating = Recmodel.getUsersRating_t(users.cuda())
        else:
            rating = Recmodel.getUsersRating_s(users.cuda())
        exclude_index = []
        exclude_items = []

        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)

        rating[exclude_index, exclude_items] = -(1<<10)
        _, rating_k = torch.topk(rating, k=max_K)

        del rating
        users_list.append(users)
        rating_list.append(rating_k)
        groundTrue_list.append(groundTrue)
        
        X = zip(rating_list, groundTrue_list)
        if multicore == 1:
            pre_results = pool.map(test_one_batch, X, config)
        else:
            pre_results = []
            for x in X:
                pre_results.append(test_one_batch(x, config))
        for result in pre_results:
            results['recall'] += result['recall']
            results['precision'] += result['precision']
            results['ndcg'] += result['ndcg']
            results['hr'] += result['hr']

        results['recall'] /= float(len(users))
        results['precision'] /= float(len(users))
        results['ndcg'] /= float(len(users))
        if multicore == 1:
            pool.close()
        return results

import pdb
import math
import pandas as pd
import numpy as np
import random
import os
import torch
import scipy.stats


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(42)

def FilterDataframeByList(target_data, col, col_name):
    df_col = pd.DataFrame(col)
    df_col.columns = [col_name]
    output = pd.merge(target_data, df_col, how='inner', left_on=[col_name], right_on=[col_name])
    return output

    
def ToList(x):
    return list(x)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def output_result(data):
    data = np.array(data).T
    for i in range(len(data)):
        m, h = mean_confidence_interval(data[i], confidence=0.95)
        m = round(m,3)
        h = round(h,3)
        print(f"{m} ± {h}  ", end=" ")
        
class DataBank():
    def __init__(self, dict_item_id2index_t,
                dict_item_id2index_s,
                dict_user_id2index_t,
                dict_user_id2index_s,
                dict_item2vec_t,
                dict_item2vec_s,
                dict_ItemIndex2vec_t,
                dict_ItemIndex2vec_s,
                dict_interactions_t,
                dict_interactions_s,
                n_users_t,
                m_items_t,
                n_users_s,
                m_items_s):
        super(DataBank, self).__init__()
        
        self.n_users_t = n_users_t
        self.m_items_t = m_items_t
        self.n_users_s = n_users_s
        self.m_items_s = m_items_s

        self.dict_item_id2index_t = dict_item_id2index_t
        self.dict_item_id2index_s = dict_item_id2index_s
        self.dict_user_id2index_t = dict_user_id2index_t
        self.dict_user_id2index_s = dict_user_id2index_s
        self.dict_item2vec_t = dict_item2vec_t 
        self.dict_item2vec_s = dict_item2vec_s
        self.dict_ItemIndex2vec_t = dict_ItemIndex2vec_t
        self.dict_ItemIndex2vec_s = dict_ItemIndex2vec_s
        self.dict_interactions_t = dict_interactions_t
        self.dict_interactions_s = dict_interactions_s


        train_users_t, val_users_t, test_users_t = [], [], []
        train_items_t, val_items_t, test_items_t = [], [], []
        #数据集按照8:1:1进行划分
        for users in dict_interactions_t.keys():
            val_start = -(math.ceil(len(dict_interactions_t[users]) * 0.1)*2) #valid数据起始位置为总长度的最后0.2
            test_start = - math.ceil(len(dict_interactions_t[users]) * 0.1) #test数据起始位置为总长度的最后0.1
            train_users_t.extend([users] * len(dict_interactions_t[users][:val_start]))
            val_users_t.extend([users] * len(dict_interactions_t[users][val_start:test_start]))
            test_users_t.extend([users] * len(dict_interactions_t[users][test_start:]))

            train_items_t.extend(dict_interactions_t[users][:val_start])
            val_items_t.extend(dict_interactions_t[users][val_start:test_start])
            test_items_t.extend(dict_interactions_t[users][test_start:])

        self.train_users_t, self.val_users_t, self.test_users_t = np.array(train_users_t), np.array(val_users_t), np.array(test_users_t)
        self.train_items_t, self.val_items_t, self.test_items_t = np.array(train_items_t), np.array(val_items_t), np.array(test_items_t)
        
                
        train_users_s, val_users_s, test_users_s = [], [], []
        train_items_s, val_items_s, test_items_s = [], [], []
        # 数据集按照8:1:1进行划分
        for users in dict_interactions_s.keys():
            val_start = -(math.ceil(len(dict_interactions_s[users]) * 0.1)*2)
            test_start = -math.ceil(len(dict_interactions_s[users]) * 0.1)
            train_users_s.extend([users] * len(dict_interactions_s[users][:val_start]))
            val_users_s.extend([users] * len(dict_interactions_s[users][val_start:test_start]))
            test_users_s.extend([users] * len(dict_interactions_s[users][test_start:]))

            train_items_s.extend(dict_interactions_s[users][:val_start])
            val_items_s.extend(dict_interactions_s[users][val_start:test_start])
            test_items_s.extend(dict_interactions_s[users][test_start:])

        self.train_users_s, self.val_users_s, self.test_users_s = np.array(train_users_s), np.array(val_users_s), np.array(test_users_s)
        self.train_items_s, self.val_items_s, self.test_items_s = np.array(train_items_s), np.array(val_items_s), np.array(test_items_s)
        


def Evaluation(Reclist, groundtruth, t_u, K):
    hit_rite = []
    ndcg = []
    for i in K:
        hr_i = 0.
        ndcg_i = 0.
        for j in range(len(t_u)):
            hr_i += len(set(Reclist[t_u[j]][:i]) & set([groundtruth[j]]))
            ndcg_i += getNDCG(Reclist[t_u[j]][:i], [groundtruth[j]])
        hit_rite.append(hr_i / len(groundtruth))
        ndcg.append(ndcg_i / len(groundtruth))
    return hit_rite, ndcg
def getDCG(scores):
    return np.sum(
        np.divide(np.power(2, scores) - 1, np.log(np.arange(scores.shape[0], dtype=np.float32) + 2)),
        dtype=np.float32)
def getNDCG(rank_list, pos_items):
    relevance = np.ones_like(pos_items)
    it2rel = {it: r for it, r in zip(pos_items, relevance)}
    rank_scores = np.asarray([it2rel.get(it, 0.0) for it in rank_list], dtype=np.float32)

    idcg = getDCG(relevance)

    dcg = getDCG(rank_scores)

    if dcg == 0.0:
        return 0.0

    ndcg = dcg / idcg
    return ndcg
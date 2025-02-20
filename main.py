import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import datetime
import torch
from torch import nn, optim
import sys
import time
import random
import os
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import scipy.stats
from util import *
from model import *
from optimization import *
import os
from parse import parse_args

def train(Recmodel, config, dictBank):
    patient = 10
    min_ = 0.
    for epoch in range(config["epochs"]):
        if epoch % 5 == 0:
            print("")
            results = Test(Recmodel, config, dictBank.dict_interactions_t, 1, 1)
            print("[target VAL]:",results)
            results_s = Test(Recmodel, config, dictBank.dict_interactions_s, 1, 0)
            print("[source VAL]:",results_s)
            if results['ndcg'][0] > min_:
                min_ = results['ndcg'][0]
                patient = 10
                continue
            if results['ndcg'][0] <= min_:
                if Recmodel.init_c != 1:
                    print("using codebook.------------")
                    Recmodel.init_codebook()
                Recmodel.init_c = 1
                patient = patient - 1
            if patient == 0:
                break
        start = time.time()
        aver_loss, aver_pre_loss, aver_reg_loss, aver_prototype_loss = Train_on_epoch(Recmodel, config, dictBank)
        end = time.time()
        sys.stdout.write(
            "\r ||epoch:{0}||loss:{1}||pre_loss:{2}||reg_loss:{3}||prototype_loss:{4}||time:{5}".format(epoch,
                                                                                                                aver_loss,
                                                                                                                aver_pre_loss,
                                                                                                                aver_reg_loss,
                                                                                                                aver_prototype_loss,
                                                                                                                round(
                                                                                                                    end - start,
                                                                                                                    2)))
        sys.stdout.flush()
    print("Training Done.")


args = parse_args()
config = vars(args)

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device_cuda = torch.device('cuda')
target_platform = args.target_platform
source_platform = args.source_platform
data_path = './Datasets/'

source_file = f"Rating_{source_platform}asS.csv"
target_file = f"Rating_{target_platform}asT.csv"

print("Target platform file:" + target_file)
print("Source platform file:" + source_file)

df_S = pd.read_csv(data_path + source_file)
df_T = pd.read_csv(data_path + target_file)

model_path = f"./Models/"
dict_path = './Datasets/'

# source platform
dict_item2vec_s = np.load(f'{dict_path}/Dict_item2vec_{source_platform}_LLM.npy',
                        allow_pickle=True).item()
vec_matrix_s = [dict_item2vec_s[movieId] for movieId in df_S.movieId.unique()]
dict_ItemIndex2vec_s = dict(zip(np.arange(len(df_S.movieId.unique())), vec_matrix_s))
dict_item_id2index_s = dict(zip(df_S.movieId.unique(), np.arange(len(df_S.movieId.unique()))))
dict_user_id2index_s = dict(zip(df_S.userId.unique(), np.arange(len(df_S.userId.unique()))))
df_S["account_index"] = df_S.userId.map(lambda x: dict_user_id2index_s[x])
df_S["deal_index"] = df_S.movieId.map(lambda x: dict_item_id2index_s[x])
dict_interactions_s = dict(df_S.groupby(df_S["account_index"])["deal_index"].apply(ToList))
n_users_s = len(df_S.account_index.unique())
m_items_s = len(df_S.deal_index.unique())
n_inters_s = df_S.shape[0]
print(f"source n_users:{n_users_s}, m_items:{m_items_s}, n=inter.:{n_inters_s}")

# target platform
dict_item2vec_t = np.load(f'{dict_path}/Dict_item2vec_{target_platform}_LLM.npy',
                            allow_pickle=True).item()
vec_matrix_t = [dict_item2vec_t[movieId] for movieId in df_T.movieId.unique()]
dict_ItemIndex2vec_t = dict(zip(np.arange(len(df_T.movieId.unique())), vec_matrix_t))
dict_item_id2index_t = dict(zip(df_T.movieId.unique(), np.arange(len(df_T.movieId.unique()))))
dict_user_id2index_t = dict(zip(df_T.userId.unique(), np.arange(len(df_T.userId.unique()))))
df_T["account_index"] = df_T.userId.map(lambda x: dict_user_id2index_t[x])
df_T["deal_index"] = df_T.movieId.map(lambda x: dict_item_id2index_t[x])
dict_interactions_t = dict(df_T.groupby(df_T["account_index"])["deal_index"].apply(ToList))
n_users_t = len(df_T.account_index.unique())
m_items_t = len(df_T.deal_index.unique())
n_inters_t = df_T.shape[0]
print(f"target n_users:{n_users_t}, m_items:{m_items_t}, n=inter.:{n_inters_t}")

my_dictBank = DataBank(dict_item_id2index_t,
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
                       m_items_s
                       )


UINet_s = csr_matrix((np.ones(len(my_dictBank.train_users_s)), (my_dictBank.train_users_s, my_dictBank.train_items_s)), shape=(n_users_s, m_items_s))
UINet_t = csr_matrix((np.ones(len(my_dictBank.train_users_t)), (my_dictBank.train_users_t, my_dictBank.train_items_t)), shape=(n_users_t, m_items_t))

print(config)
weight_file = model_path + f"{config['model']}_{source_platform}to{target_platform}.pth.tar"
if config["pretrained_model"]:
    Recmodel = twolgn(UINet_t, UINet_s, my_dictBank, config).to(device_cuda)
    Recmodel.load_state_dict(torch.load(weight_file))
else:
    Recmodel = twolgn(UINet_t, UINet_s, my_dictBank, config).to(device_cuda)
    train(Recmodel, config, my_dictBank)
    # torch.save(Recmodel.state_dict(), weight_file)

print(f"task is :{source_platform} -> {target_platform}")
print("config:", config)
print("The final res:")
results = Test(Recmodel, config, my_dictBank.dict_interactions_t, 0, 1)
print(f"target {target_platform}:", results)
results_s = Test(Recmodel, config, my_dictBank.dict_interactions_s, 0, 0)
print(f"source {source_platform}:",results_s)

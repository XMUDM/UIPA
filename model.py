import numpy as np
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from torch import nn, optim
import sys
import time
import os
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from util import *
from prototype import vq, vq_st
import datetime


class VQEmbedding(nn.Module):
    def __init__(self, K, D):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)

    def re_assign(self, embedding_weight):
        self.embedding.weight = embedding_weight

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.contiguous()

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
                                               dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.contiguous()

        return z_q_x, z_q_x_bar


class twolgn(nn.Module):
    def __init__(self, UINet_t, UINet_s, dictBank, config: dict):
        super(twolgn, self).__init__()

        self.config = config

        self.num_Users_t = dictBank.n_users_t
        self.num_Items_t = dictBank.m_items_t

        self.num_Users_s = dictBank.n_users_s
        self.num_Items_s = dictBank.m_items_s

        self.latent_dim = self.config['latent_dim_rec']

        self.n_layers = self.config['lightGCN_n_layers']

        self.dictBank = dictBank

        self.prototype_num = config['prototype_num']

        # init weight
        self.embedding_user_t = torch.nn.Embedding(
            num_embeddings=self.num_Users_t, embedding_dim=self.latent_dim).cuda()
        self.embedding_user_s = torch.nn.Embedding(
            num_embeddings=self.num_Users_s, embedding_dim=self.latent_dim).cuda()

        self.fc_i = nn.Linear(768, self.latent_dim).cuda()
        self.fc_map = nn.Linear(self.latent_dim, self.latent_dim)

        self.f = nn.Sigmoid()
        self.UIGraph_t = getGraph(self.num_Users_t, self.num_Items_t, UINet_t).cuda()
        self.UIGraph_s = getGraph(self.num_Users_s, self.num_Items_s, UINet_s).cuda()

        self.users_gap = []
        self.users_map = []

        self.codebook_t = VQEmbedding(self.prototype_num, self.latent_dim)
        self.codebook_s = VQEmbedding(self.prototype_num, self.latent_dim)

        self.codebook_t_item = VQEmbedding(self.prototype_num, self.latent_dim)
        self.codebook_s_item = VQEmbedding(self.prototype_num, self.latent_dim)

        self.encoder_t = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.encoder_s = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder_s = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder_t = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder_s_item = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder_t_item = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

        self.id_embedding_item_t = torch.nn.Embedding(
            num_embeddings=self.num_Items_t, embedding_dim=self.latent_dim).cuda()
        self.id_embedding_item_s = torch.nn.Embedding(
            num_embeddings=self.num_Items_s, embedding_dim=self.latent_dim).cuda()

        self.init_weights()
        self.text_embedding_item_t = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(list(self.dictBank.dict_ItemIndex2vec_t.values())), freeze=True).cuda()
        self.text_embedding_item_s = torch.nn.Embedding.from_pretrained(
            torch.FloatTensor(list(self.dictBank.dict_ItemIndex2vec_s.values())), freeze=True).cuda()
        self.tau = 1
        self.init_c = 0
        self.init_pseudo_s = 0
        self.init_pseudo_t = 0
        self.dropout = nn.Dropout(p=0.2)

        self.weight = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)]).cuda()
        self.bi_weight = nn.ModuleList(
            [nn.Linear(self.latent_dim, self.latent_dim) for _ in range(self.n_layers)]).cuda()

        self.fusion_item = nn.Linear(self.latent_dim * 2, self.latent_dim)

        self.item_fusion_weight = nn.Parameter(torch.tensor(0.5))
        print(f"using {config['model']}")
        model_ = {'lightgcn': self.lightgcn_forward, 'MF': self.MF_forward, 'NGCF': self.NGCF_forward}
        self.model_forward = model_[config['model']]

        self.user_num_s = np.zeros(len(self.codebook_s.embedding.weight), dtype=int)
        self.user_num_t = np.zeros(len(self.codebook_t.embedding.weight), dtype=int)
        self.user_2_code_index_s = np.zeros(self.num_Users_s, dtype=int)
        self.user_2_code_index_t = np.zeros(self.num_Users_t, dtype=int)


        self.pseudo_user_list_s = [[] for _ in range(len(self.codebook_s.embedding.weight))]
        self.pseudo_user_list_t = [[] for _ in range(len(self.codebook_t.embedding.weight))]

        self.batch_i = -1

        self.zero_code_user_list_t = [[] for _ in range(len(self.codebook_t.embedding.weight))]
        self.zero_code_user_list_s = [[] for _ in range(len(self.codebook_s.embedding.weight))]


        self.zero_code_bpr_user_t = [[] for _ in range(len(self.codebook_t.embedding.weight))]
        self.zero_code_bpr_user_s = [[] for _ in range(len(self.codebook_s.embedding.weight))]

        self.zero_code_bpr_posI_t = [[] for _ in range(len(self.codebook_t.embedding.weight))]
        self.zero_code_bpr_posI_s = [[] for _ in range(len(self.codebook_s.embedding.weight))]

        self.zero_code_bpr_negI_t = [[] for _ in range(len(self.codebook_t.embedding.weight))]
        self.zero_code_bpr_negI_s = [[] for _ in range(len(self.codebook_s.embedding.weight))]


        self.users_embedding_all_t_time = torch.nn.Embedding(
            num_embeddings=self.num_Users_t, embedding_dim=self.latent_dim).cuda()
        self.users_embedding_all_s_time = torch.nn.Embedding(
            num_embeddings=self.num_Users_s, embedding_dim=self.latent_dim).cuda()

        self.users_embedding_all_t_time.weight.data = self.embedding_user_t.weight.data.detach().clone()
        self.users_embedding_all_s_time.weight.data = self.embedding_user_s.weight.data.detach().clone()


        self.embedding_user_t_time = torch.nn.Embedding(
            num_embeddings=self.num_Users_t, embedding_dim=self.latent_dim).cuda()
        self.embedding_user_s_time = torch.nn.Embedding(
            num_embeddings=self.num_Users_s, embedding_dim=self.latent_dim).cuda()
        self.codebook_t_time = VQEmbedding(self.prototype_num, self.latent_dim)
        self.codebook_s_time = VQEmbedding(self.prototype_num, self.latent_dim)

        self.embedding_user_s_time.weight.data = self.embedding_user_s.weight.data.detach().clone()
        self.embedding_user_t_time.weight.data = self.embedding_user_t.weight.data.detach().clone()
        self.codebook_s_time.embedding.weight.data = self.codebook_s.embedding.weight.data.detach().clone()
        self.codebook_t_time.embedding.weight.data = self.codebook_t.embedding.weight.data.detach().clone()


    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.1)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.1)
                nn.init.normal_(module.bias, std=0.1)

    def init_codebook(self):
        users_t, items_t, _, _, _, _ = self.getEmbedding(1)
        users_s, items_s, _, _, _, _ = self.getEmbedding(0)
        embedding_user = torch.cat((users_t, users_s), dim=0).cpu().detach().numpy()
        embedding_item = torch.cat((items_t, items_s), dim=0).cpu().detach().numpy()

        kmedoids = KMedoids(n_clusters=self.prototype_num, random_state=0)
        clusters = kmedoids.fit_predict(embedding_user)
        centers = kmedoids.cluster_centers_
        self.codebook_t.embedding.weight.data = torch.tensor(centers, requires_grad=True).cuda()
        self.codebook_s.embedding.weight.data = torch.tensor(centers, requires_grad=True).cuda()

        kmedoids_item = KMedoids(n_clusters=self.prototype_num, random_state=0)
        clusters_item = kmedoids_item.fit_predict(embedding_item)
        centers_item = kmedoids_item.cluster_centers_
        self.codebook_t_item.embedding.weight.data = torch.tensor(centers_item, requires_grad=True).cuda()
        self.codebook_s_item.embedding.weight.data = torch.tensor(centers_item, requires_grad=True).cuda()

        self.get_code_user_num()
        self.get_zero_code_user_list()

    def item_fusion(self, item_id, item_text, fusion_mode="weight_add"):
        if fusion_mode == "add":
            users_emb = (item_id + item_text) / 2.
        elif fusion_mode == "weight_add":
            users_emb = item_id * (1 - self.config['llm_weight']) + item_text * self.config['llm_weight']
        elif fusion_mode == "learn_weight_add":
            users_emb = item_id * torch.sigmoid(self.item_fusion_weight) + item_text * (
                        1 - torch.sigmoid(self.item_fusion_weight))
        elif fusion_mode == "attention":
            attention_scores = torch.matmul(item_id, item_text.T)
            global_to_local_weights = F.softmax(attention_scores, dim=1)

            weighted_item_text = torch.matmul(global_to_local_weights.T, item_text)

            attention_scores = torch.matmul(item_text, item_id.T)
            local_to_global_weights = F.softmax(attention_scores, dim=1)

            weighted_item_id = torch.matmul(local_to_global_weights.T, item_id)

            users_emb = (weighted_item_id + weighted_item_text) / 2.
        elif fusion_mode == "concat":
            users_emb = torch.cat((item_text, item_id), 1)
            users_emb = self.fusion_item(users_emb)

        return users_emb

    def lightgcn_forward(self, users_emb, items_emb, UIGraph):

        num_Users = users_emb.shape[0]
        num_Items = items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for i in range(self.n_layers):
            all_emb = torch.sparse.mm(UIGraph, all_emb)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_Users, num_Items])
        return users, items

    def MF_forward(self, users_emb, items_emb, UIGraph):

        users, items = users_emb, items_emb
        return users, items

    def NGCF_forward(self, users_emb, items_emb, UIGraph):
        num_Users = users_emb.shape[0]
        num_Items = items_emb.shape[0]
        all_emb = torch.cat([users_emb, items_emb])
        embs = [all_emb]
        for layer in range(self.n_layers):
            side_emb = torch.sparse.mm(UIGraph, all_emb)
            sum_emb = self.weight[layer](side_emb)
            bi_emb = torch.mul(all_emb, side_emb)
            bi_emb = self.bi_weight[layer](bi_emb)
            all_emb = nn.LeakyReLU(negative_slope=0.2)(sum_emb + bi_emb)
            all_emb = F.normalize(all_emb, p=2, dim=1)
            embs.append(all_emb)

        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [num_Users, num_Items])
        return users, items

    def sim(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return torch.mm(z1, z2.t())

    def contrastive_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.tau)
        between_sim = f(self.sim(z1, z2))  # [B, N]
        losses = -torch.log(between_sim.diag() / between_sim.sum(1))
        return losses.mean()

    def getEmbedding(self, is_target):
        if is_target:
            embedding_item = self.id_embedding_item_t.weight
            embedding_user = self.embedding_user_t.weight
            UIGraph = self.UIGraph_t
            codebook = self.codebook_t
            decoder = self.decoder_t
            codebook_item = self.codebook_t_item
            decoder_item = self.decoder_t_item
        else:
            embedding_item = self.id_embedding_item_s.weight
            embedding_user = self.embedding_user_s.weight
            UIGraph = self.UIGraph_s
            codebook = self.codebook_s
            decoder = self.decoder_s
            codebook_item = self.codebook_s_item
            decoder_item = self.decoder_s_item

        users, items = self.model_forward(embedding_user, embedding_item, UIGraph)

        if is_target:
            text_embedding_item = self.text_embedding_item_t.weight
        else:
            text_embedding_item = self.text_embedding_item_s.weight
        text_embedding_item = self.fc_i(text_embedding_item)
        items = self.item_fusion(items, text_embedding_item)

        _, user_2_codebook = codebook.straight_through(users)
        _, item_2_codebook = codebook_item.straight_through(items)
        if self.init_c:
            users = users * (1 - self.config['prototype_weight']) + decoder(user_2_codebook) * self.config['prototype_weight']
            items = items * (1 - self.config['prototype_weight']) + decoder_item(item_2_codebook) * self.config['prototype_weight']

        return users, items, embedding_user, embedding_item, user_2_codebook, item_2_codebook

    def get_codebook_loss_user(self, codebook_user, user_2_codebook, is_target):
        if self.init_c == 0:
            return torch.tensor(0.0, requires_grad=True)

        vq_loss = F.mse_loss(user_2_codebook, codebook_user.detach())

        commit_loss = F.mse_loss(codebook_user, user_2_codebook.detach())
        codebook_loss = vq_loss + commit_loss

        if is_target:
            codebook_align_loss = self.contrastive_loss(self.codebook_t.embedding.weight,
                                                        self.codebook_s.embedding.weight.detach())
        else:
            codebook_align_loss = self.contrastive_loss(self.codebook_s.embedding.weight,
                                                        self.codebook_t.embedding.weight.detach())

        codebook_loss += codebook_align_loss

        return codebook_loss

    def get_codebook_loss_item(self, codebook_item, item_2_codebook, is_target):
        if self.init_c == 0:
            return torch.tensor(0.0, requires_grad=True)

        vq_loss = F.mse_loss(item_2_codebook, codebook_item.detach())

        commit_loss = F.mse_loss(codebook_item, item_2_codebook.detach())
        codebook_loss = vq_loss + commit_loss

        if is_target:
            codebook_align_loss = self.contrastive_loss(self.codebook_t_item.embedding.weight,
                                                        self.codebook_s_item.embedding.weight.detach())
        else:
            codebook_align_loss = self.contrastive_loss(self.codebook_s_item.embedding.weight,
                                                        self.codebook_t_item.embedding.weight.detach())

        codebook_loss += codebook_align_loss

        return codebook_loss

    def getLoss(self, users, posI, negI, is_target):

        users_embedding_all, items_embedding_all, users_embedding_ego, items_embedding_ego, user_2_codebook, item_2_codebook = self.getEmbedding(
            is_target)

        if self.init_c == 1:
            if is_target:
                users_embedding_all_time = self.users_embedding_all_t_time.weight
                zero_code_bpr_user = self.zero_code_bpr_user_t
                zero_code_bpr_posI = self.zero_code_bpr_posI_t
                zero_code_bpr_negI = self.zero_code_bpr_negI_t
            else:
                users_embedding_all_time = self.users_embedding_all_s_time.weight
                zero_code_bpr_user = self.zero_code_bpr_user_s
                zero_code_bpr_posI = self.zero_code_bpr_posI_s
                zero_code_bpr_negI = self.zero_code_bpr_negI_s

            for i in range(len(zero_code_bpr_user)):
                if len(zero_code_bpr_user[i]):

                    users_embedding_time = users_embedding_all_time[torch.tensor(zero_code_bpr_user[i], dtype=int)]
                    pos_items_embedding_time = items_embedding_all[torch.tensor(zero_code_bpr_posI[i], dtype=int)]
                    neg_items_embedding_time = items_embedding_all[torch.tensor(zero_code_bpr_negI[i], dtype=int)]
                    pos_scores_time = torch.mul(users_embedding_time, pos_items_embedding_time)
                    pos_scores_time = torch.sum(pos_scores_time, dim=1)
                    neg_scores_time = torch.mul(users_embedding_time, neg_items_embedding_time)
                    neg_scores_time = torch.sum(neg_scores_time, dim=1)
                    bpr_loss_time = torch.mean(
                        torch.nn.functional.softplus(neg_scores_time - pos_scores_time)).cpu().detach().item()

                    users_embedding_time_plus_1 = users_embedding_all[torch.tensor(zero_code_bpr_user[i], dtype=int)]
                    pos_items_embedding_time_plus_1 = items_embedding_all[torch.tensor(zero_code_bpr_posI[i], dtype=int)]
                    neg_items_embedding_time_plus_1 = items_embedding_all[torch.tensor(zero_code_bpr_negI[i], dtype=int)]
                    pos_scores_time_plus_1 = torch.mul(users_embedding_time_plus_1, pos_items_embedding_time_plus_1)
                    pos_scores_time_plus_1 = torch.sum(pos_scores_time_plus_1, dim=1)
                    neg_scores_time_plus_1 = torch.mul(users_embedding_time_plus_1, neg_items_embedding_time_plus_1)
                    neg_scores_time_plus_1 = torch.sum(neg_scores_time_plus_1, dim=1)
                    bpr_loss_time_plus_1 = torch.mean(
                        torch.nn.functional.softplus(neg_scores_time_plus_1 - pos_scores_time_plus_1)).cpu().detach().item()

                    if bpr_loss_time_plus_1 > bpr_loss_time:

                        if is_target:
                            self.embedding_user_t.weight.data[self.zero_code_user_list_t[i]] = \
                            self.embedding_user_t_time.weight.data[self.zero_code_user_list_t[i]].detach().clone()
                            self.codebook_t.embedding.weight.data[i] = self.codebook_t_time.embedding.weight.data[i].detach().clone()
                        else:
                            self.embedding_user_s.weight.data[self.zero_code_user_list_s[i]] = \
                            self.embedding_user_s_time.weight.data[self.zero_code_user_list_s[i]].detach().clone()
                            self.codebook_s.embedding.weight.data[i] = self.codebook_s_time.embedding.weight.data[i].detach().clone()

            users_embedding_all, items_embedding_all, users_embedding_ego, items_embedding_ego, user_2_codebook, item_2_codebook = self.getEmbedding(
                is_target)

        users_embedding = users_embedding_all[users]
        pos_items_embedding = items_embedding_all[posI]
        neg_items_embedding = items_embedding_all[negI]

        num_batch = len(users)

        users_emb_ego = users_embedding_ego[users]
        pos_emb_ego = items_embedding_ego[posI]
        neg_emb_ego = items_embedding_ego[negI]


        reg_loss = (1 / 2) * (
                    users_emb_ego.norm(2).pow(2) + pos_emb_ego.norm(2).pow(2) + neg_emb_ego.norm(2).pow(2)) / num_batch

        for module in self.modules():
            if isinstance(module, nn.Linear):
                reg_loss += (1 / 2) * (module.weight.norm(2).pow(2) +
                                       module.bias.norm(2).pow(2))

        pos_scores = torch.mul(users_embedding, pos_items_embedding)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_embedding, neg_items_embedding)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        if self.init_c == 1:
            if self.batch_i == 0:
                self.get_pseudo_user_list()

            pseudo_code_loss = self.get_zero_codebook_loss()
            codebook_loss_user = self.get_codebook_loss_user(users_embedding_all[users], user_2_codebook[users],
                                                   is_target)
            codebook_loss_item = self.get_codebook_loss_item(items_embedding_all[posI], item_2_codebook[posI],
                                                   is_target)

            codebook_loss = codebook_loss_item + codebook_loss_user + pseudo_code_loss
        else:
            codebook_loss = torch.tensor(0.0, requires_grad=True)

        if is_target == 1:
            self.users_embedding_all_t_time.weight.data = users_embedding_all.detach().clone()
            self.embedding_user_t_time.weight.data = self.embedding_user_t.weight.data.detach().clone()
            self.codebook_t_time.embedding.weight.data = self.codebook_t.embedding.weight.data.detach().clone()

        if is_target == 0:
            self.users_embedding_all_s_time.weight.data = users_embedding_all.detach().clone()
            self.embedding_user_s_time.weight.data = self.embedding_user_s.weight.data.detach().clone()
            self.codebook_s_time.embedding.weight.data = self.codebook_s.embedding.weight.data.detach().clone()



        return loss, reg_loss, codebook_loss

    def forward(self, users, items):
        users_embedding_all, items_embedding_all, users_embedding_ego, items_embedding_ego, user_2_codebook, item_2_codebook = self.getEmbedding(
            is_target=1)

        users_emb = users_embedding_all[users.long()]
        items_emb = items_embedding_all[items.long()]

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating.cpu()

    def getUsersRating_t(self, users):
        users_embedding_all, items_embedding_all, users_embedding_ego, items_embedding_ego, user_2_codebook, item_2_codebook = self.getEmbedding(
            is_target=1)

        users_emb = users_embedding_all[users.long()]
        items_emb = items_embedding_all

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating.cpu()

    def getUsersRating_s(self, users):
        users_embedding_all, items_embedding_all, users_embedding_ego, items_embedding_ego, user_2_codebook, item_2_codebook = self.getEmbedding(
            is_target=0)

        users_emb = users_embedding_all[users.long()]
        items_emb = items_embedding_all

        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating.cpu()

    def getCodeInfo_user(self):
        users_embedding_all_s, items_embedding_all_s, _, _, user_2_codebook_s, item_2_codebook_s = self.getEmbedding(is_target=0)
        users_embedding_all_t, items_embedding_all_t, _, _, user_2_codebook_t, item_2_codebook_t = self.getEmbedding(is_target=1)
        users_embedding_all_s = users_embedding_all_s.cpu().detach().numpy()
        users_embedding_all_t = users_embedding_all_t.cpu().detach().numpy()
        user_2_codebook_s = user_2_codebook_s.cpu().detach().numpy()
        user_2_codebook_t = user_2_codebook_t.cpu().detach().numpy()
        codebook_s = self.codebook_s.embedding.weight.cpu().detach().numpy()
        codebook_t = self.codebook_t.embedding.weight.cpu().detach().numpy()
        return users_embedding_all_s, users_embedding_all_t, user_2_codebook_s, user_2_codebook_t, codebook_s, codebook_t

    def get_code_user_num(self):
        users_embedding_all_s, users_embedding_all_t, user_2_codebook_s, user_2_codebook_t, codebook_s, codebook_t = self.getCodeInfo_user()  # 数据都是numpy.ndarray类型
        for i in range(len(user_2_codebook_s)):
            for j in range(len(codebook_s)):
                if (user_2_codebook_s[i] == codebook_s[j]).all():
                    self.user_num_s[j] += 1
                    self.user_2_code_index_s[i] = j
        for i in range(len(user_2_codebook_t)):
            for j in range(len(codebook_t)):
                if (user_2_codebook_t[i] == codebook_t[j]).all():
                    self.user_num_t[j] += 1
                    self.user_2_code_index_t[i] = j


    def get_zero_code_user_list(self):
        index_s = np.arange(len(self.user_2_code_index_s))
        index_t = np.arange(len(self.user_2_code_index_t))
        for i in range(len(self.user_num_s)):
            if self.user_num_s[i] == 0 and self.user_num_t[i] != 0:
                self.zero_code_user_list_t[i] = index_t[self.user_2_code_index_t == i]
            if self.user_num_s[i] != 0 and self.user_num_t[i] == 0:
                self.zero_code_user_list_s[i] = index_s[self.user_2_code_index_s == i]


    def get_pseudo_user_list(self):
        users_embedding_all_s, _, _, _, _, _ = self.getEmbedding(is_target=0)
        users_embedding_all_t, _, _, _, _, _ = self.getEmbedding(is_target=1)
        for i in range(len(self.user_num_s)):
            if self.user_num_s[i] == 0 and self.user_num_t[i] != 0:
                self.pseudo_user_list_s[i] = []
                user_emb = users_embedding_all_t[self.zero_code_user_list_t[i]]
                for j in range(len(user_emb)):
                    mse_loss = torch.mean(F.mse_loss(user_emb[j].repeat(len(users_embedding_all_s), 1), users_embedding_all_s, reduction='none'), dim=1)
                    self.pseudo_user_list_s[i].extend(
                        torch.topk(mse_loss, k=1, largest=False).indices.cpu().detach().tolist())
            if self.user_num_s[i] != 0 and self.user_num_t[i] == 0:
                self.pseudo_user_list_t[i] = []
                user_emb = users_embedding_all_s[self.zero_code_user_list_s[i]]

                for j in range(len(user_emb)):
                    mse_loss = torch.mean(F.mse_loss(user_emb[j].repeat(len(users_embedding_all_t), 1), users_embedding_all_t, reduction='none'), dim=1)
                    self.pseudo_user_list_t[i].extend(
                        torch.topk(mse_loss, k=1, largest=False).indices.cpu().detach().tolist())


    def get_zero_codebook_loss(self):
        users_embedding_all_s, _, _, _, _, _ = self.getEmbedding(is_target=0)
        users_embedding_all_t, _, _, _, _, _ = self.getEmbedding(is_target=1)
        zero_code_loss = 0.
        for i in range(len(self.user_num_s)):
            if self.user_num_s[i] == 0 and self.user_num_t[i] != 0:
                pseudo_user_emb = users_embedding_all_s[self.pseudo_user_list_s[i]]
                pseudo_code_emb = torch.mean(pseudo_user_emb, dim=0).detach()
                zero_code_loss += F.mse_loss(self.codebook_t.embedding.weight[i], pseudo_code_emb.detach())
            if self.user_num_s[i] != 0 and self.user_num_t[i] == 0:
                pseudo_user_emb = users_embedding_all_t[self.pseudo_user_list_t[i]]
                pseudo_code_emb = torch.mean(pseudo_user_emb, dim=0).detach()
                zero_code_loss += F.mse_loss(self.codebook_s.embedding.weight[i], pseudo_code_emb.detach())
        return zero_code_loss

    def init_pseudo_info(self, all_users, posItems, negItems, is_target):
        if is_target == 1:
            for i in range(len(self.zero_code_user_list_t)):
                if len(self.zero_code_user_list_t[i]):
                    user_set = set(self.zero_code_user_list_t[i])
                    all_users_index = [x for x, y in enumerate(all_users.cpu().detach().numpy().tolist()) if
                                       y in user_set]
                    user_2_posItems = posItems[all_users_index].cpu().detach().numpy().tolist()
                    user_2_negItems = negItems[all_users_index].cpu().detach().numpy().tolist()
                    all_users_index = all_users[all_users_index].cpu().detach().numpy().tolist()
                    self.zero_code_bpr_user_t[i].extend(all_users_index)
                    self.zero_code_bpr_posI_t[i].extend(user_2_posItems)
                    self.zero_code_bpr_negI_t[i].extend(user_2_negItems)
        else:
            for i in range(len(self.zero_code_user_list_s)):
                if len(self.zero_code_user_list_s[i]):
                    user_set = set(self.zero_code_user_list_s[i])
                    all_users_index = [x for x, y in enumerate(all_users.cpu().detach().numpy().tolist()) if
                                       y in user_set]
                    user_2_posItems = posItems[all_users_index].cpu().detach().numpy().tolist()
                    user_2_negItems = negItems[all_users_index].cpu().detach().numpy().tolist()
                    all_users_index = all_users[all_users_index].cpu().detach().numpy().tolist()
                    self.zero_code_bpr_user_s[i].extend(all_users_index)
                    self.zero_code_bpr_posI_s[i].extend(user_2_posItems)
                    self.zero_code_bpr_negI_s[i].extend(user_2_negItems)


def getGraph(n_users, m_items, Net):
    adj_mat = sp.dok_matrix((n_users + m_items, n_users + m_items), dtype=np.float32)
    adj_mat = adj_mat.tolil()
    R = Net.tolil()
    adj_mat[:n_users, n_users:] = R
    adj_mat[n_users:, :n_users] = R.T
    adj_mat = adj_mat.todok()

    rowsum = np.array(adj_mat.sum(axis=1))
    d_inv = np.power(rowsum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)

    norm_adj = d_mat.dot(adj_mat).dot(d_mat).tocsr()
    coo = norm_adj.tocoo().astype(np.float32)
    index = torch.stack([torch.LongTensor(coo.row), torch.LongTensor(coo.col)])
    data = torch.FloatTensor(coo.data)
    Graph = torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    return Graph.coalesce()

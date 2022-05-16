#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : model.py
# @Author: yanms
# @Date  : 2021/11/1 16:16
# @Desc  : fuse with learning weight
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from gcn_conv import GCNConv
from utils import BPRLoss, EmbLoss


class GraphEncoder(nn.Module):
    def __init__(self, layers, hidden_dim):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, add_self_loops=False, cached=False) for i in range(layers)])


    def forward(self, x, edge_index):

        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index)
        return x


class CRGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(CRGCN, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.dropout = args.dropout
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors = args.behaviors
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers, self.embedding_size) for behavior in self.behaviors
        })

        self.layer_weight = {
            behavior: torch.ones(self.n_users + self.n_items + 2, self.embedding_size) for behavior in self.behaviors
        }

        self.reg_weight = args.reg_weight
        self.layers = args.layers
        self.dropout = args.dropout
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters)
        else:
            if isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight.data)


    def gcn_propagate(self):
        """
        gcn propagate in each behavior
        """
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        for behavior in self.behaviors:
            layer_embeddings = total_embeddings
            indices = self.edge_index[behavior].to(self.device)
            layer_embeddings = self.Graph_encoder[behavior](layer_embeddings, indices)
            layer_embeddings = F.normalize(layer_embeddings, dim=-1)
            total_embeddings = layer_embeddings * self.layer_weight[behavior] + total_embeddings
            all_embeddings[behavior] = total_embeddings
        return all_embeddings

    def forward(self, batch_data):
        self.storage_all_embeddings = None

        all_embeddings = self.gcn_propagate()
        total_loss = 0
        for index, behavior in enumerate(self.behaviors):
            data = batch_data[:, index]
            users = data[:, 0].long()
            items = data[:, 1:].long()
            user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
            user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
            item_feature = item_all_embedding[items]
            scores = torch.sum(user_feature * item_feature, dim=2)
            total_loss += self.bpr_loss(scores[:, 0], scores[:, 1])
        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight, self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]], [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))
        return scores


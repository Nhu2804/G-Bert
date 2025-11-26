from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import copy
import json
import math
import logging
import numpy as np

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from torch.nn import LayerNorm
import torch.nn.functional as F
from config import BertConfig
from bert_models import BERT, PreTrainedBertModel, BertLMPredictionHead, TransformerBlock, gelu
import dill

logger = logging.getLogger(__name__)

CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


def freeze_afterwards(model):
    for p in model.parameters():
        p.requires_grad = False


class TSNE(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, proc_voc=None):  # ĐỔI: rx_voc → proc_voc
        super(TSNE, self).__init__(config)

        self.bert = BERT(config, dx_voc, proc_voc)  # ĐỔI: rx_voc → proc_voc
        self.dx_voc = dx_voc
        self.proc_voc = proc_voc  # ĐỔI: rx_voc → proc_voc

        freeze_afterwards(self)

    def forward(self, output_dir, output_file='graph_embedding.tsv'):
        if not self.config.graph:
            print('save embedding not graph')
            proc_graph_emb = self.bert.embedding.word_embeddings(  # ĐỔI: rx_graph_emb → proc_graph_emb
                torch.arange(3, len(self.proc_voc.word2idx) + 3, dtype=torch.long))  # ĐỔI: rx_voc → proc_voc
            dx_graph_emb = self.bert.embedding.word_embeddings(
                torch.arange(len(self.proc_voc.word2idx) + 3, len(self.proc_voc.word2idx) + 3 + len(self.dx_voc.word2idx),  # ĐỔI: rx_voc → proc_voc
                             dtype=torch.long))
        else:
            print('save embedding graph')

            dx_graph_emb = self.bert.embedding.ontology_embedding.dx_embedding.get_all_graph_emb()
            proc_graph_emb = self.bert.embedding.ontology_embedding.proc_embedding.get_all_graph_emb()  # ĐỔI: rx_embedding → proc_embedding

        np.savetxt(os.path.join(output_dir, 'dx-' + output_file),
                   dx_graph_emb.detach().numpy(), delimiter='\t')
        np.savetxt(os.path.join(output_dir, 'proc-' + output_file),  # ĐỔI: rx → proc
                   proc_graph_emb.detach().numpy(), delimiter='\t')

        # def dump(prefix='dx-', emb):
        #     with open(prefix + output_file ,'w') as fout:
        #         m = emb.detach().cpu().numpy()
        #         for
        #         fout.write()


class ClsHead(nn.Module):
    def __init__(self, config: BertConfig, voc_size):
        super(ClsHead, self).__init__()
        self.cls = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.ReLU(
        ), nn.Linear(config.hidden_size, voc_size))

    def forward(self, input):
        return self.cls(input)


class SelfSupervisedHead(nn.Module):
    def __init__(self, config: BertConfig, dx_voc_size, proc_voc_size):  # ĐỔI: rx_voc_size → proc_voc_size
        super(SelfSupervisedHead, self).__init__()
        self.multi_cls = nn.ModuleList([ClsHead(config, dx_voc_size), ClsHead(
            config, dx_voc_size), ClsHead(config, proc_voc_size), ClsHead(config, proc_voc_size)])  # ĐỔI: rx_voc_size → proc_voc_size

    def forward(self, dx_inputs, proc_inputs):  # ĐỔI: rx_inputs → proc_inputs
        # inputs (B, hidden)
        # output logits
        return self.multi_cls[0](dx_inputs), self.multi_cls[1](proc_inputs), self.multi_cls[2](dx_inputs), self.multi_cls[3](proc_inputs)  # ĐỔI: rx_inputs → proc_inputs


class GBERT_Pretrain(PreTrainedBertModel):
    def __init__(self, config: BertConfig, dx_voc=None, proc_voc=None):  # ĐỔI: rx_voc → proc_voc
        super(GBERT_Pretrain, self).__init__(config)
        self.dx_voc_size = len(dx_voc.word2idx)
        self.proc_voc_size = len(proc_voc.word2idx)  # ĐỔI: rx_voc_size → proc_voc_size

        self.bert = BERT(config, dx_voc, proc_voc)  # ĐỔI: rx_voc → proc_voc
        self.cls = SelfSupervisedHead(
            config, self.dx_voc_size, self.proc_voc_size)  # ĐỔI: rx_voc_size → proc_voc_size

        self.apply(self.init_bert_weights)

    def forward(self, inputs, dx_labels=None, proc_labels=None):  # ĐỔI: rx_labels → proc_labels
        # inputs (B, 2, max_len)
        # bert_pool (B, hidden)
        _, dx_bert_pool = self.bert(inputs[:, 0, :], torch.zeros(
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))
        _, proc_bert_pool = self.bert(inputs[:, 1, :], torch.zeros(  # ĐỔI: rx_bert_pool → proc_bert_pool
            (inputs.size(0), inputs.size(2))).long().to(inputs.device))

        dx2dx, proc2dx, dx2proc, proc2proc = self.cls(dx_bert_pool, proc_bert_pool)  # ĐỔI: rx2dx → proc2dx, dx2rx → dx2proc, rx2rx → proc2proc
        # output logits
        if proc_labels is None or dx_labels is None:  # ĐỔI: rx_labels → proc_labels
            return F.sigmoid(dx2dx), F.sigmoid(proc2dx), F.sigmoid(dx2proc), F.sigmoid(proc2proc)  # ĐỔI: rx2dx → proc2dx, dx2rx → dx2proc, rx2rx → proc2proc
        else:
            loss = F.binary_cross_entropy_with_logits(dx2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(proc2dx, dx_labels) + \
                F.binary_cross_entropy_with_logits(dx2proc, proc_labels) + \
                F.binary_cross_entropy_with_logits(proc2proc, proc_labels)  # ĐỔI: rx2rx → proc2proc, rx_labels → proc_labels
            return loss, F.sigmoid(dx2dx), F.sigmoid(proc2dx), F.sigmoid(dx2proc), F.sigmoid(proc2proc)  # ĐỔI: rx2dx → proc2dx, dx2rx → dx2proc, rx2rx → proc2proc


class MappingHead(nn.Module):
    def __init__(self, config: BertConfig):
        super(MappingHead, self).__init__()
        self.dense = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                   nn.ReLU())

    def forward(self, input):
        return self.dense(input)


class GBERT_Predict(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer):
        super(GBERT_Predict, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.proc_voc)  # ĐỔI: rx_voc → proc_voc
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.proc_voc_multi.word2idx)))  # ĐỔI: rx_voc_multi → proc_voc_multi

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, proc_labels=None, epoch=None):  # ĐỔI: rx_labels → proc_labels
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param proc_labels: [adm-1, proc_size]  # ĐỔI: rx_labels → proc_labels
        :param dx_labels: [adm-1, dx_size]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        proc_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)  # ĐỔI: rx_bert_pool → proc_bert_pool

        # mean and concat for procedure prediction task  # ĐỔI: rx prediction → procedure prediction
        proc_logits = []  # ĐỔI: rx_logits → proc_logits
        for i in range(proc_labels.size(0)):  # ĐỔI: rx_labels → proc_labels
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            proc_mean = torch.mean(proc_bert_pool[0:i+1, :], dim=0, keepdim=True)  # ĐỔI: rx_mean → proc_mean
            # concat
            concat = torch.cat(
                [dx_mean, proc_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)  # ĐỔI: rx_mean → proc_mean
            proc_logits.append(self.cls(concat))  # ĐỔI: rx_logits → proc_logits

        proc_logits = torch.cat(proc_logits, dim=0)  # ĐỔI: rx_logits → proc_logits
        loss = F.binary_cross_entropy_with_logits(proc_logits, proc_labels)  # ĐỔI: rx_logits → proc_logits, rx_labels → proc_labels
        return loss, proc_logits  # ĐỔI: rx_logits → proc_logits


class GBERT_Predict_Side(PreTrainedBertModel):
    def __init__(self, config: BertConfig, tokenizer, side_len):
        super(GBERT_Predict_Side, self).__init__(config)
        self.bert = BERT(config, tokenizer.dx_voc, tokenizer.proc_voc)  # ĐỔI: rx_voc → proc_voc
        self.dense = nn.ModuleList([MappingHead(config), MappingHead(config)])
        self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, 2*config.hidden_size),
                                 nn.ReLU(), nn.Linear(2*config.hidden_size, len(tokenizer.proc_voc_multi.word2idx)))  # ĐỔI: rx_voc_multi → proc_voc_multi

        self.side = nn.Sequential(nn.Linear(
            side_len, side_len // 2), nn.ReLU(), nn.Linear(side_len // 2, side_len // 2))
        self.final_cls = nn.Sequential(nn.ReLU(), nn.Linear(len(
            tokenizer.proc_voc_multi.word2idx) + side_len // 2, len(tokenizer.proc_voc_multi.word2idx)))  # ĐỔI: rx_voc_multi → proc_voc_multi
        # self.cls = nn.Sequential(nn.Linear(3*config.hidden_size, len(tokenizer.proc_voc_multi.word2idx)))  # ĐỔI: rx_voc_multi → proc_voc_multi
        # self.gru = nn.GRU(config.hidden_size, config.hidden_size, batch_first=True)

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, dx_labels=None, proc_labels=None, epoch=None, input_sides=None):  # ĐỔI: rx_labels → proc_labels
        """
        :param input_ids: [B, max_seq_len] where B = 2*adm
        :param proc_labels: [adm-1, proc_size]  # ĐỔI: rx_labels → proc_labels
        :param dx_labels: [adm-1, dx_size]
        :param input_side: [adm-1, side_len]
        :return:
        """
        token_types_ids = torch.cat([torch.zeros((1, input_ids.size(1))), torch.ones(
            (1, input_ids.size(1)))], dim=0).long().to(input_ids.device)
        token_types_ids = token_types_ids.repeat(
            1 if input_ids.size(0)//2 == 0 else input_ids.size(0)//2, 1)
        # bert_pool: (2*adm, H)
        _, bert_pool = self.bert(input_ids, token_types_ids)
        loss = 0
        bert_pool = bert_pool.view(2, -1, bert_pool.size(1))  # (2, adm, H)
        dx_bert_pool = self.dense[0](bert_pool[0])  # (adm, H)
        proc_bert_pool = self.dense[1](bert_pool[1])  # (adm, H)  # ĐỔI: rx_bert_pool → proc_bert_pool

        # mean and concat for procedure prediction task  # ĐỔI: rx prediction → procedure prediction
        visit_vecs = []
        for i in range(proc_labels.size(0)):  # ĐỔI: rx_labels → proc_labels
            # mean
            dx_mean = torch.mean(dx_bert_pool[0:i+1, :], dim=0, keepdim=True)
            proc_mean = torch.mean(proc_bert_pool[0:i+1, :], dim=0, keepdim=True)  # ĐỔI: rx_mean → proc_mean
            # concat
            concat = torch.cat(
                [dx_mean, proc_mean, dx_bert_pool[i+1, :].unsqueeze(dim=0)], dim=-1)  # ĐỔI: rx_mean → proc_mean
            concat_trans = self.cls(concat)
            visit_vecs.append(concat_trans)

        visit_vecs = torch.cat(visit_vecs, dim=0)
        # add side and concat
        side_trans = self.side(input_sides)
        patient_vec = torch.cat([visit_vecs, side_trans], dim=1)

        proc_logits = self.final_cls(patient_vec)  # ĐỔI: rx_logits → proc_logits
        loss = F.binary_cross_entropy_with_logits(proc_logits, proc_labels)  # ĐỔI: rx_logits → proc_logits, rx_labels → proc_labels
        return loss, proc_logits  # ĐỔI: rx_logits → proc_logits

# ------------------------------------------------------------

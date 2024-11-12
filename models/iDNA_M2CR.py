import torch
import random
import numpy as np
import torch.nn as nn
from models import DNABERT, FusionNet

import config_init
from components import DataManager, ModelManager, IOManager, Trainer


class iDNA_M2CR(nn.Module):
    def __init__(self, config, len_seqs: int):
        super(iDNA_M2CR, self).__init__()
        self.config = config
        self.num_views = len(config.feature_dims)
        for _, k in enumerate(config.kmers):
            if k == 3:  # 3mer
                self.config.kmer = k
                self.bert_3mer = DNABERT.BERT(self.config)
            elif k == 4:  # 4mer
                self.config.kmer = k
                self.bert_4mer = DNABERT.BERT(self.config)
            elif k == 5:  # 5mer
                self.config.kmer = k
                self.bert_5mer = DNABERT.BERT(self.config)
            else:  # 6mer
                self.config.kmer = k
                self.bert_6mer = DNABERT.BERT(self.config)

        self.FusionNet = FusionNet.FusionNet(self.config, len_seqs)

    def get_representations(self, seqs):
        representations = dict()
        for idx, k in enumerate(self.config.kmers):
            if k == 3:
                representations[idx] = self.bert_3mer(seqs).to('cuda' if self.config.cuda else 'cpu')
            elif k == 4:
                representations[idx] = self.bert_4mer(seqs).to('cuda' if self.config.cuda else 'cpu')
            elif k == 5:
                representations[idx] = self.bert_5mer(seqs).to('cuda' if self.config.cuda else 'cpu')
            else:
                representations[idx] = self.bert_6mer(seqs).to('cuda' if self.config.cuda else 'cpu')
        return representations

    def pre_train(self, seqs, labels: torch.Tensor, idx: torch.Tensor):
        representations = self.get_representations(seqs)
        pre_train_loss = self.FusionNet.dec_part(representations, labels, idx)
        return pre_train_loss

    def forward(self, seqs):
        representations = self.get_representations(seqs)
        output, fusion_representation = self.FusionNet(representations)
        return output, fusion_representation

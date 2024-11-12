import torch.nn as nn
from typing import List
from transformers import BertTokenizer, BertConfig, BertModel


class BERT(nn.Module):
    def __init__(self, config):
        super(BERT, self).__init__()
        self.config = config

        self.k = config.kmer
        if self.k == 3:
            self.pretrainpath = './fine_tuned_model/4mC/3/DNABERT_3mer'
        elif self.k == 4:
            self.pretrainpath = './fine_tuned_model/4mC/4/DNABERT_4mer'
        elif self.k == 5:
            self.pretrainpath = './fine_tuned_model/4mC/5/DNABERT_5mer'
        else:
            self.pretrainpath = './fine_tuned_model/4mC/6/DNABERT_6mer'

        self.setting = BertConfig.from_pretrained(self.pretrainpath, num_labels=2, finetuning_task="dnaprom", cache_dir=None)
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrainpath)
        self.bert = BertModel.from_pretrained(self.pretrainpath, config=self.setting)

    def forward(self, seqs: List[str]):
        k = self.config.kmer
        kmer = [[seqs[i][x: x + k] for x in range(len(seqs[i]) - k + 1)] for i in range(len(seqs))]
        kmers = [" ".join(kmer[i]) for i in range(len(kmer))]
        # input tokenizer and get tensors return
        token_seq = self.tokenizer(kmers, return_tensors='pt')
        input_ids, token_type_ids, attention_mask = token_seq['input_ids'], token_seq['token_type_ids'], token_seq['attention_mask']
        if self.config.cuda:
            representation = self.bert(input_ids.cuda(), token_type_ids.cuda(), attention_mask.cuda())['pooler_output']
        else:
            representation = self.bert(input_ids, token_type_ids, attention_mask)['pooler_output']
        return representation


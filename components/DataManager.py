import torch
import torch.utils.data as Data
from util import data_preprocess
from sklearn.model_selection import StratifiedKFold

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class DataManager():
    def __init__(self, trainer):
        self.trainer = trainer
        self.IOM = trainer.IOM
        self.config = trainer.config
        if self.config.mode == 'independent test' or 'cross-species':
            self.train_label, self.test_label = None, None
            self.train_data, self.test_data = None, None
            self.train_dataloader, self.test_dataloader = None, None
        elif self.config.mode == '10-CV':
            self.data, self.label = None, None
            self.train_dataloader, self.valid_dataloader = dict(), dict()
        else:
            self.IOM.log.Error('No Such Mode')

    def load_data(self):
        if self.config.mode == 'independent test' or 'cross-species':
            self.train_data, self.train_label = data_preprocess.load_fasta_format_data(self.config.path_train_data)
            self.test_data, self.test_label = data_preprocess.load_fasta_format_data(self.config.path_test_data)
            self.train_dataloader = self.construct_dataloader(self.train_data, self.train_label, self.config.batch_size)
            self.test_dataloader = self.construct_dataloader(self.test_data, self.test_label, self.config.batch_size)
        elif self.config.mode == '10-CV':
            self.data, self.label = data_preprocess.load_fasta_format_data(self.config.path_train_data)
            skfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=self.config.seed)
            k = 0
            for train_idx, valid_idx in skfold.split(self.data, self.label):
                self.train_dataloader[k] = self.construct_dataloader([self.data[idx] for idx in train_idx],
                                                                     [self.label[idx] for idx in train_idx], self.config.batch_size)
                self.valid_dataloader[k] = self.construct_dataloader([self.data[idx] for idx in valid_idx],
                                                                     [self.label[idx] for idx in valid_idx], self.config.batch_size)
                k += 1
        else:
            self.IOM.log.Error('No Such Mode')

    def construct_dataloader(self, data, labels, batch_size):
        if self.config.cuda:
            labels = torch.cuda.LongTensor(labels)
        else:
            labels = torch.LongTensor(labels)
        dataset = MyDataSet(data, labels)
        dataloader = Data.DataLoader(dataset, batch_size=batch_size, drop_last=False, shuffle=True)
        return dataloader

    def get_train_numbers(self):
        if self.config.mode == 'independent test' or 'cross-species':
            return len(self.train_data)
        elif self.config.mode == '10-CV':
            return len(self.data)
        else:
            self.IOM.log.Error('No Such Mode')


class MyDataSet(Data.Dataset):
    def __init__(self, data, label):
        super().__init__()
        self.data, self.label, self.device = data, label, device

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index], self.label[index], index

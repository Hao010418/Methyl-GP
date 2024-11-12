import time
import os
import pickle
import torch
import numpy as np


class IOManager():
    def __init__(self, trainer):
        self.trainer = trainer
        self.config = trainer.config

        self.result_path = None
        self.log = None

    def initialize(self):
        self.result_path = self.config.model_save_path + "_" + str(self.config.kmers) + "-mer"
        # create path to save model
        if not os.path.exists(self.result_path):
            os.makedirs(self.result_path)

        # generate .pkl doc
        with open(self.result_path + '/config.pkl', 'wb') as file:
            pickle.dump(self.config, file)

        # generate .txt doc
        with open(self.result_path + '/config.txt', 'w') as f:
            for key, value in self.config.__dict__.items():
                key_value_pair = '{}: {}'.format(key, value)
                f.write(key_value_pair + '\r\n')

        # generate log doc for saving experiments results
        self.log = LOG(self.result_path)

    # To save the best model params
    def save_model_dict(self, model_dict, model_save_name, metric_name, metric_value):
        filename = f'{model_save_name}_{metric_name}[{metric_value:.4f}].pt'
        save_path_pt = os.path.join(self.result_path, filename)
        torch.save(model_dict, save_path_pt, _use_new_zipfile_serialization=False)

    def save_predict_data(self, logit, prob, label):
        save_path = self.result_path
        logit = logit.cpu().detach().numpy()
        prob = prob.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        np.save(save_path + '/predict_data.npy', logit)
        np.save(save_path + '/predict_prob.npy', prob)
        np.save(save_path + '/predict_label.npy', label)


class LOG():
    def __init__(self, root_path):
        log_path = root_path + '/log'
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        self.log = open(log_path + '/%s.txt' % time.strftime("%Y_%m_%d_%I_%M_%S"), 'w+')
        log_files = os.listdir(log_path)
        log_files.sort()
        if len(log_files) > 200:
            print('log file >200, delete old file', log_files.pop(0), file=self.log)

    def Info(self, *data):
        msg = time.strftime("%Y-%m-%d_%I:%M:%S") + " INFO: "
        for info in data:
            if type(info) is int:
                msg = msg + str(info)
            else:
                msg = msg + str(info)
        print(msg)
        print(msg, file=self.log)

    def Warn(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " WARN: "
        for info in data:
            if type(info) is int:
                msg = msg + str(info)
            else:
                msg = msg + info
        print(msg)
        print(msg, file=self.log)

    def Error(self, *data):
        msg = time.strftime("%Y-%M-%d_%I:%M:%S") + " ERROR: "
        for info in data:
            if type(info) is int:
                msg = msg + str(info)
            else:
                msg = msg + info
        print(msg)
        print(msg, file=self.log)

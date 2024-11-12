import os
import random
import torch
import numpy as np

import config_init
from components import Trainer


def train_and_test(config):
    # for independent test
    trainer = Trainer.Trainer(config)
    trainer.setIO()
    trainer.load_data()
    trainer.init_model()
    if not config.froze:
        trainer.check_model()
    trainer.choose_loss_function()
    trainer.train_model()


def train_and_valid(config):
    # for 10-CV
    trainer = Trainer.Trainer(config)
    trainer.setIO()
    trainer.load_data()
    trainer.choose_loss_function()
    trainer.train_model()


def seed_torch(seed=42):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)  # prohibit hash
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)  # if use multiple GPUs
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


if __name__ == '__main__':
    from transformers import logging
    logging.set_verbosity_warning()
    logging.set_verbosity_error()

    config = config_init.get_config()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device)
    seed_torch(config.seed)
    dataset_name = config.dataset_name
    species = config.species
    if config.mode == 'independent test':
        path = config.methylation_type
        config.path_train_data = f'{dataset_name}/{path}/{species}/train.txt'
        config.path_test_data = f'{dataset_name}/{path}/{species}/test.txt'
        config.model_save_path = config.model_save_path + path + '/' + species
        train_and_test(config)
    else:   # config.mode == 'cross-species'
        mode = config.mode
        path = config.methylation_type
        test_species = config.test_species
        config.path_train_data = f'{dataset_name}/{path}/{species}/train.txt'
        config.path_test_data = f'{dataset_name}/{path}/{test_species}/test.txt'
        config.model_save_path = config.model_save_path + config.mode + '/' + f'{species}_{test_species}'
        train_and_test(config)

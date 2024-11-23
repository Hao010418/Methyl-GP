import torch
import torch.nn as nn
import numpy as np

from models import Methyl_GP
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score


class ModelManager():
    def __init__(self, trainer):
        self.trainer = trainer
        self.IOM = trainer.IOM
        self.DM = trainer.DM
        self.config = trainer.config
        self.device = 'cuda' if torch.cuda.is_available() and self.config.cuda else 'cpu'

        self.num_views = len(self.config.feature_dims)
        self.mode = self.config.mode
        self.loss_function = None

        self.best_performance = None
        self.best_logit_list = None
        self.best_rep_list = None
        self.best_label_list = None
        self.test_performance = []
        self.valid_performance = []
        self.avg_pretrain_losses = []
        self.avg_train_losses = []
        self.avg_test_losses = []
        # for 10-CV
        self.acc_all = []
        self.sn_all = []
        self.sp_all = []
        self.auc_all = []
        self.mcc_all = []

    def init_model(self):
        if self.mode == 'independent test' or '10-CV' or 'cross-species':
            self.model = Methyl_GP.Methy_GP(self.config, self.DM.get_train_numbers())
        else:
            self.IOM.log.Error('No Such Mode')

    def check_model(self):  # check params of model
        if self.config.check_params:
            print('-' * 50, 'Model.named_parameters', '-' * 50)
            for name, value in self.model.named_parameters():
                print(f'[{name}]' + '-' * 10 + '>' + f'[{value.shape}], [requires_grad: {value.requires_grad}]')
        if self.config.count_params:
            num_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print('=' * 50, "Number of total parameters:" + str(num_params), '=' * 50)

    def choose_loss_function(self):
        if self.config.loss_function == 'CE':
            self.loss_function = nn.CrossEntropyLoss()
        else:
            self.IOM.log.Error('No Such Loss Function')

    def get_loss(self, logits: torch.Tensor, label: torch.Tensor, idx, representation, epoch=50):
        w_c = dict()
        for k, w in enumerate(self.model.FusionNet.W):
            w_c[k] = w.abs().mean(dim=-1)
        w_transform = self.model.FusionNet.sparsepart(w_c)
        loss_clf, loss_bn_w, l1_loss_bn = 0, 0, 0
        loss_clf = self.loss_function(logits, label)
        for k in range(self.num_views):
            last_bn = None
            for layer in self.model.FusionNet.encoder.enc[k]:
                if isinstance(layer, nn.BatchNorm1d):
                    last_bn = layer
            loss_bn_w += torch.mean((last_bn.weight - w_transform[k]) ** 2)
            l1_loss_bn += last_bn.weight.abs().sum()
        loss_bn_w /= self.num_views
        loss_a = torch.sum((representation - self.model.FusionNet.emb(idx)) ** 2, dim=-1) + self.config.sigma * loss_bn_w
        loss = torch.mean(
            self.config.lambda_1 * loss_clf + max(0, 1 - epoch / 10) * loss_a) + self.config.lambda_2 * l1_loss_bn
        loss = (loss - self.config.b).abs() + self.config.b
        return loss

    def froze_params(self, model):
        for name, param in model.named_parameters():
            if 'bert' in name:
                param.requires_grad = False

    def train_part(self):
        if self.mode == 'independent test' or self.mode == 'cross-species':
            self.model.train()
            train_dataloader, test_dataloader = self.DM.train_dataloader, self.DM.test_dataloader
            model_params, performance, rep, logit, prob, label, roc_data, prc_data = self.train(train_dataloader, test_dataloader)
            if self.config.save_model:
                self.IOM.save_model_dict(model_params, self.config.model_save_name, 'ACC', performance[0])
            if self.config.save_pred_data:
                self.IOM.save_predict_data(logit, prob, label)

            self.best_logit_list = logit
            self.best_rep_list = rep
            self.best_label_list = label
            self.best_performance = performance
            self.IOM.log.Info('Best Performance: {}'.format(self.best_performance))
        else:
            self.IOM.log.Error('No Such Mode')

    def train(self, train_dataloader, test_dataloader):
        best_acc = 0
        best_model_params, best_performance, best_ROC, best_PRC = None, None, None, None
        best_logit, best_rep, best_prob, best_label = None, None, None, None
        self.model.to(self.device)
        # froze bert or not
        if self.config.froze:
            self.froze_params(model=self.model)
            self.check_model()
        # pre-train part for fusion net
        optimizer = torch.optim.SGD(self.model.FusionNet.parameters(), lr=0.00005, weight_decay=0.003)
        for epoch in range(1, self.config.pre_epoch + 1):
            self.model.train()
            pretrain_losses = []
            for data, label, idx in train_dataloader:
                idx = idx.to(self.device)
                loss = self.model.pre_train(data, label, idx)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                pretrain_losses.append(loss.item())
            avg_pretrain_loss = np.average(pretrain_losses)
            self.avg_pretrain_losses.append(avg_pretrain_loss)
            print(f'Epoch {epoch:3d}: average pretrain loss {avg_pretrain_loss:.6f}')

        # fine-tuning part
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.wd)
        self.choose_loss_function()
        for epoch in range(1, self.config.epoch + 1):
            self.model.train()
            correct, num_samples = 0, 0
            train_losses, valid_losses = [], []
            for data, label, idx in train_dataloader:
                idx = idx.to(self.device)
                logit, representation = self.model(data)
                loss = self.get_loss(logit, label, idx, representation)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())
                num_samples += len(label)
                correct += torch.sum(logit.argmax(dim=-1).eq(label)).item()
            train_acc = correct / num_samples
            avg_train_loss = np.average(train_losses)
            self.avg_train_losses.append(avg_train_loss)
            # validation
            self.model.eval()
            test_performance, rep_list, logit_list, prob_list, label_list, ROC_data, PRC_data, avg_test_loss = self.test(test_dataloader)
            self.test_performance.append(test_performance)
            test_acc = test_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
            self.avg_test_losses.append(avg_test_loss)
            print(f'Train results {epoch}/{self.config.epoch}: loss: {avg_train_loss:.5f}, ACC: {train_acc:.4f}')
            log_text = '\n' + '-' * 20 + f'Test Performance Epoch[{epoch}/{self.config.epoch}]' + '-' * 20 \
                       + '\n[ACC,\tSN,\t\tSP,\t\tAUC,\tMCC]' + (f'\n{test_performance[0]:.4f},\t{test_performance[1]:.4f},'
                        f'\t{test_performance[2]:.4f},\t{test_performance[3]:.4f},\t{test_performance[4]:.4f}\n')
            self.IOM.log.Info(log_text)
            # record best ACC
            if test_acc > best_acc:
                best_acc = test_acc
                best_performance = test_performance
                best_ROC = ROC_data
                best_PRC = PRC_data
                best_logit = logit_list
                best_rep = rep_list
                best_prob = prob_list
                best_label = label_list
                best_model_params = self.model.state_dict()

        return best_model_params, best_performance, best_rep, best_logit, best_prob, best_label, best_ROC, best_PRC

    def test(self, test_dataloader):
        test_loss, cnt = 0, 0,
        prob_list, pred_list = torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device)
        logit_list, rep_list, label_list = torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device), torch.Tensor([]).to(self.device)
        avg_test_loss = 0

        with torch.no_grad():
            for data, label, idx in test_dataloader:
                logit, representation = self.model(data)
                prob, pred = logit.softmax(dim=-1), logit.argmax(dim=-1)
                avg_test_loss += self.loss_function(logit, label)

                logit_list = torch.cat((logit_list, logit))  # logits for drawing UMAP pictures
                rep_list = torch.cat((rep_list, representation))
                label_list = torch.cat((label_list, label))
                prob_list = torch.cat((prob_list, prob))
                pred_list = torch.cat((pred_list, pred))  # label, prob and pred for drawing AUC and PR curve
                cnt += 1

        avg_test_loss /= cnt
        performance, ROC_data, PRC_data = self.caculate_metrics(prob_list[:, 1], pred_list, label_list)
        return performance, rep_list, logit_list, prob_list, label_list, ROC_data, PRC_data, avg_test_loss

    def caculate_metrics(self, prob, pred, label):
        prob = prob.cpu().numpy()
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        num_samples = len(label)
        tp, fp, tn, fn = 0, 0, 0, 0
        for i in range(num_samples):
            if label[i] == 1:
                if pred[i] == 1:
                    tp += 1
                else:
                    fn += 1
            else:
                if pred[i] == 0:
                    tn += 1
                else:
                    fp += 1
        ACC = float(tp + tn) / num_samples

        # compute Sensitivity
        if tp + fn == 0:
            Recall = Sensitivity = 0
        else:
            Recall = Sensitivity = float(tp) / (tp + fn)
        # compute Specificity
        if tn + fp == 0:
            Specificity = 0
        else:
            Specificity = float(tn) / (tn + fp)
        # compute MCC
        if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
            MCC = 0
        else:
            MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
        # compute ROC and AUC
        fpr, tpr, thresholds = roc_curve(label, prob, pos_label=1)
        AUC = auc(fpr, tpr)
        # compute PRC and AP
        precision, recall, thresholds = precision_recall_curve(label, prob, pos_label=1)
        AP = average_precision_score(label, prob, average='macro', pos_label=1, sample_weight=None)

        performance = [ACC, Sensitivity, Specificity, AUC, MCC]
        roc_data = [fpr, tpr, AUC]
        prc_data = [recall, precision, AP]
        return performance, roc_data, prc_data

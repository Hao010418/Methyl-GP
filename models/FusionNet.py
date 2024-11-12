import torch
import torch.nn as nn
from typing import List


class Decoder(nn.Module):
    def __init__(self, feature_dims: List[int], fc_hidden: int):
        super(Decoder, self).__init__()
        self.dec = nn.ModuleList([
            nn.Sequential(nn.Linear(fc_hidden, feature_dim), nn.ReLU()) for feature_dim in feature_dims
        ])

    def forward(self, x: dict):
        y = dict()
        for k, v in x.items():
            y[k] = self.dec[k](v)
        return y


class Encoder(nn.Module):
    def __init__(self, feature_dims: List[int], fc_hidden: int, embedding_dim: int):
        super(Encoder, self).__init__()
        self.enc = nn.ModuleList([
            nn.Sequential(nn.Linear(feature_dim, fc_hidden), nn.BatchNorm1d(fc_hidden), nn.ReLU(),
                          nn.Linear(fc_hidden, embedding_dim), nn.BatchNorm1d(embedding_dim), nn.ReLU()
                          ) for feature_dim in feature_dims
        ])

    def forward(self, x: dict):
        embedding = dict()
        for k, v in x.items():
            embedding[k] = self.enc[k](v)
        return embedding


class ElementLinear(nn.Module):
    def __init__(self, embedding_dim: int):
        super(ElementLinear, self).__init__()
        self.weight = nn.Parameter(torch.rand(embedding_dim))
        self.bias = nn.Parameter(torch.rand(embedding_dim))

    def forward(self, x: torch.Tensor):
        return x * self.weight + self.bias


class SparseConstraint(nn.Module):
    def __init__(self, num_views: int, embedding_dim: int):
        super(SparseConstraint, self).__init__()
        self.spc = nn.ModuleList([
            nn.Sequential(ElementLinear(embedding_dim), nn.Sigmoid()) for _ in range(num_views)
        ])

    def forward(self, x: dict):
        output = dict()
        for k, v in x.items():
            output[k] = self.spc[k](v)
        return output


class FusionNet(nn.Module):
    def __init__(self, config, n_train: int):
        super(FusionNet, self).__init__()
        self.config = config
        self.n_train = n_train

        self.feature_dims = config.feature_dims
        self.num_views = len(config.feature_dims)
        self.emb_dim = config.emb_dim
        self.fc_hidden = config.fc_hidden
        # initial weights of decoder part
        self.W = nn.ParameterList([nn.Parameter(torch.rand(self.emb_dim, self.fc_hidden)) for _ in self.feature_dims])
        self.emb = nn.Embedding(self.n_train, self.emb_dim)
        nn.init.normal_(self.emb.weight.data)
        self.decoder = Decoder(self.feature_dims, self.fc_hidden)
        # initial weights of encoder part
        self.encoder = Encoder(self.feature_dims, self.fc_hidden, embedding_dim=self.emb_dim)
        self.sparsepart = SparseConstraint(self.num_views, embedding_dim=self.emb_dim)
        self.classifier = nn.Sequential(nn.Linear(self.emb_dim, 20), nn.Dropout(0.5), nn.ReLU(), nn.Linear(20, 2))

    def dec_part(self, inputs: dict, label: torch.Tensor, idx: torch.Tensor):
        """decoder part"""
        # get corresponding samples' embedding
        z_hat = self.emb(idx)
        views = dict()
        for k in range(self.num_views):
            views[k] = torch.einsum('ij, jk->ik', z_hat, self.W[k])
        input_hat = self.decoder(views)

        # calculate the reconstruction loss
        loss_r, loss_w = torch.zeros_like(label).float(), torch.tensor(0.0, device=label.device)
        for k in range(self.num_views):
            loss_r += torch.mean((input_hat[k] - inputs[k]) ** 2, dim=-1)
            # loss_w += torch.mean(self.W[k] ** 2, dim=-1).sum()
            loss_w += torch.sqrt(torch.sum(self.W[k] ** 2, dim=-1)).sum()
        return torch.mean(loss_r) + self.config.beta * loss_w

    def forward(self, inputs: dict):
        """encoder part"""
        views = self.encoder(inputs)
        # get fusion representation of all views
        fusion_representation = sum(views.values()) / self.num_views
        ouputs = self.classifier(fusion_representation)

        return ouputs, fusion_representation

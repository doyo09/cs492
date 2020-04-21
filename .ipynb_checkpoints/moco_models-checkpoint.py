import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class MoCoV2(nn.Module):
    def __init__(self, base_encoder, similarity_dim=128, q_size=65536, momentum=0.999, temperature=0.07, ):
        super(MoCoV2, self).__init__()
        self.K = q_size
        self.m = momentum
        self.T = temperature

        self.q_enc = base_encoder()  # torchvision.models.__dict__['resnet50']
        self.k_enc = base_encoder()  # torchvision.models.__dict__['resnet50']

        # for mlp
        in_features = self.q_enc.fc.weight.size(1)
        self.q_enc.fc = nn.Sequential(nn.Linear(in_features, in_features), nn.ReLU(),
                                      nn.Linear(in_features=in_features, out_features=similarity_dim))
        self.k_enc.fc = nn.Sequential(nn.Linear(in_features, in_features), nn.ReLU(),
                                      nn.Linear(in_features=in_features, out_features=similarity_dim))

        # initialize k_enc params
        for q, k in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            k.data.copy_(q.data)
            k.requires_grad = False

        # initialize dynamic queue : (simil_dim,K)
        self.register_buffer("queue", torch.randn(similarity_dim, self.K))
        self.register_buffer("queue_pointer", torch.tensor(0, dtype=torch.long))

        self.queue = F.normalize(self.queue, dim=0)

    def forward(self, img_q, img_k):
        """
        no need to shuffle
        @param img_q : from moco_dataloader (bs, 3, 224, 224)
        @param img_k : from moco_dataloader (bs, 3, 224, 224)
        """
        self.momentum_update()

        simil_vec_q = self.q_enc(img_q)  # (bs, similarity_dim)
        simil_vec_q = F.normalize(simil_vec_q, dim=1)

        simil_vec_k = self.k_enc(img_k)  # (bs, similarity_dim)
        simil_vec_k = F.normalize(simil_vec_k, dim=1)
        # no grad to key
        simil_vec_k = simil_vec_k.detach()

        # positive logits : (bs,1) = (bs, 1, simil_dim) * (bs, simil_dim, 1)
        l_pos = torch.bmm(simil_vec_k.unsqueeze(1), simil_vec_k.unsqueeze(2))

        # negative logits : (bs,K) = (bs, simil_dim) * (simil_dim, K)
        l_neg = torch.mm(simil_vec_k, self.queue)

        # output, y
        logits = torch.cat([l_pos.squeeze(2), l_neg], dim=1, )
        logits /= self.T
        labels = torch.zeros(logits.size(0))  # (bs,)

        self.replace_queue_with(simil_vec_k)

        return logits, labels

    def momentum_update(self, ):
        for q_param, k_param in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            k_param.data = self.m * k_param.data + (1 - self.m) * q_param

    def replace_queue_with(self, simil_vec_k):
        """
        self.queue : (simil_dim,K)
        simil_vec_k : (bs, simil_dim)
        """
        bs = simil_vec_k.size(0)
        self.queue[:, self.queue_pointer:self.queue_pointer + bs] = simil_vec_k.T
        self.queue_pointer = (self.queue_pointer + bs) % self.K

    
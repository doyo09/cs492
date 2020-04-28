import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, std=0.001)
        nn.init.constant_(m.bias.data, 0.0)

class MoCoV2(nn.Module):
    def __init__(self, base_encoder, similarity_dim=128, q_size=128*64, momentum=0.999, temperature=0.07, ):
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
        self.q_enc.apply(weights_init_classifier)

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
        simil_vec_q = self.q_enc(img_q)  # (bs, similarity_dim)
        simil_vec_q = F.normalize(simil_vec_q, dim=1)

        with torch.no_grad():
            self.momentum_update()

            simil_vec_k = self.k_enc(img_k)  # (bs, similarity_dim)
            simil_vec_k = F.normalize(simil_vec_k, dim=1)
            # no grad to key
            simil_vec_k = simil_vec_k.detach()

        # positive logits : (bs,1) = (bs, 1, simil_dim) * (bs, simil_dim, 1)
        l_pos = torch.bmm(simil_vec_q.unsqueeze(1), simil_vec_k.unsqueeze(2))

        # negative logits : (bs,K) = (bs, simil_dim) * (simil_dim, K)
        # should be detached, since queue is buffer
        l_neg = torch.mm(simil_vec_q, self.queue.clone().detach())

        # output, y
        logits = torch.cat([l_pos.squeeze(2), l_neg], dim=1, )
        logits /= self.T
        labels = torch.zeros(logits.size(0), dtype = torch.long )  # (bs,)

        self.replace_queue_with(simil_vec_k)

        return logits, labels


    def momentum_update(self, ):
        for q_param, k_param in zip(self.q_enc.parameters(), self.k_enc.parameters()):
            k_param.data = self.m * k_param.data + (1 - self.m) * q_param.data


    def replace_queue_with(self, simil_vec_k):
        """
        self.queue : (simil_dim,K)
        simil_vec_k : (bs, simil_dim)
        """
        bs = simil_vec_k.size(0)
        self.queue[:, self.queue_pointer:self.queue_pointer + bs] = simil_vec_k.T
        self.queue_pointer = (self.queue_pointer + bs) % self.K


class Resnet50(nn.Module):
    # baseline for 첫번째 비교
    def __init__(self, base_encoder, num_classes = 265):
        super(Resnet50, self).__init__()

        self.model_ft = base_encoder(pretrained=True)  # torchvision.models.__dict__['resnet50']

        # same architecture as moco
        in_features = self.model_ft.fc.weight.size(1)
        self.model_ft.fc = nn.Sequential(nn.Linear(in_features=in_features, out_features=in_features),
                                         nn.ReLU(),
                                         nn.Linear(in_features=in_features, out_features=num_classes)
                                         )
        self.model_ft.apply(weights_init_classifier)

    def forward(self, img):
        """
        @param img : from SupervisedImageLoader (bs, 3, 224, 224)
        """
        logits = self.model_ft(img)
        return logits

# for experiment1
class LinearProtocol(nn.Module):
    def __init__(self, input_dim = 2048, class_num=265,):  # 512
        super(LinearProtocol, self).__init__()
        self.fc = nn.Linear(in_features= input_dim, out_features= class_num)
        self.fc.apply(weights_init_classifier)
    def forward(self, x) :
        return self.fc(x)

# for building the whole model
class ClassifierBlock(nn.Module):
    """
    mixmatch에 사용할 모델입니다.
    """
    def __init__(self, input_dim=2048, class_num=265, dropout=True, relu=True, num_bottleneck=512):  # 512
        super(ClassifierBlock, self).__init__()
        add_block = []
        add_block += [nn.Linear(input_dim, num_bottleneck)]
        add_block += [nn.BatchNorm1d(num_bottleneck)]
        if relu:
            add_block += [nn.ReLU()]
        if dropout:
            add_block += [nn.Dropout(p=0.5)]
        add_block = nn.Sequential(*add_block)
        # add_block.apply(weights_init_kaiming)
        classifier = []
        classifier += [nn.Linear(num_bottleneck, class_num)]
        classifier = nn.Sequential(*classifier)
        # classifier.apply(weights_init_classifier)
        self.add_block = add_block
        self.classifier = classifier
        self.add_block.apply(weights_init_classifier)
        self.classifier.apply(weights_init_classifier)
    def forward(self, x):
        x = self.add_block(x)
        x = self.classifier(x)
        return x

# for experiment1 & building the whole model
class MoCoClassifier(nn.Module):
    def __init__(self, moco_model, classifier, use_bn = False, pretrained = True, init= True):
        """
        @param moco_model : moco_model.q_enc + moco_model.k_enc
        @param classifier : Classifier block for real mix model or Linear for linear protocol
        """
        super(MoCoClassifier, self).__init__()
        self.moco_model = moco_model.q_enc
        self.classifier = classifier
        if use_bn :
            self.BN = nn.BatchNorm1d()
        if not pretrained :
            self.moco_model.apply(weights_init_classifier)
        if init :
            self.classifier.apply(weights_init_classifier)


    def forward(self, img):
        x = img
        for layer_name, layer in self.moco_model._modules.items():
            x = layer(x)
            if layer_name == "avgpool":
                break
        x = F.relu(x.view(x.size(0), -1))
        x = self.classifier(x)
        return x


class ResnetClassifier(nn.Module):
    def __init__(self, resnet, classifier, pretrained = False, init= True):
        """
        @param resnet : resnet50
        @param classifier : Classifier block for real mix model or Linear for linear protocol
        """
        super(ResnetClassifier, self).__init__()
        self.resnet = resnet.model_ft
        if not pretrained :
            self.resnet.apply(weights_init_classifier)
        self.classifier = classifier
        if init :
            self.classifier.apply(weights_init_classifier)


    def forward(self, img):
        x = img
        for layer_name, layer in self.resnet._modules.items():
            x = layer(x)
            if layer_name == "avgpool":
                break

        preds = F.relu(x.view(x.size(0), -1))
        preds = self.classifier(preds)
        return x, preds



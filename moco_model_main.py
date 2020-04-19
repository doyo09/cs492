import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np

import torch
import torch.nn as nn

from moco_dataloader import MoCoImageLoader, SupervisedImageLoader
from moco_models import MoCoV2
from utils import AverageMeter, bind_nsml

import argparse

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Pretraining MoCo')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', default=64, type=int, help='')

parser.add_argument('--print_every', default=10, type=int, help='')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
######################################################################


def split_ids(path, ratio):
    with open(path) as f:
        ids_l = []
        ids_u = []
        cnt = 0
        for i, line in enumerate(f.readlines()):
            cnt += 1
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))
    print("length of the file", cnt)
    ids_l = np.array(ids_l)
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio * len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u


def main():
    print(torch.__version__)
    global opts
    opts = parser.parse_args()

    train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
    print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
    # 비교를 위해 val_ids를 제외한다.
    # 훈련시에는 모든 unl_ids 사용
    moco_trainloader = MoCoImageLoader(DATASET_PATH, 'train', np.setdiff1d(unl_ids, val_ids),# unl_ids if you fully train moco
                                       # simCLR style transform
                                       transform=transforms.Compose([
                                           transforms.Resize(opts.imResize),
                                           transforms.RandomResizedCrop(opts.imsize),
                                           # transforms.RandomApply([
                                           #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                           #      ], p=0.8),
                                           transforms.RandomGrayscale(p=0.2),
                                           # gaussian blur should be added
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    print("len(moco_trainloader) :", len(moco_trainloader), )
    moco_trainloader = torch.utils.data.DataLoader(moco_trainloader, batch_size=opts.batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True) # num_workers = 4

    moco_valloader = MoCoImageLoader(DATASET_PATH, 'val', val_ids,
                                     # simCLR style transform
                                     transform=transforms.Compose([
                                           transforms.Resize(opts.imResize),
                                           transforms.RandomResizedCrop(opts.imsize),
                                           # transforms.RandomApply([
                                           #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                           #      ], p=0.8),
                                           transforms.RandomGrayscale(p=0.2),
                                           # gaussian blur should be added
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))


    print("len(moco_valloader) :", len(moco_valloader), )
    moco_valloader = torch.utils.data.DataLoader(moco_valloader, batch_size=opts.batch_size, shuffle=False,
                                                   pin_memory=True, drop_last=False) # num_workers = 4
    print('train_loaders done')

    ###### set device, model ######
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {d}".format(d = device))

    ###### set model ######
    moco_v2 = MoCoV2(base_encoder=models.__dict__["resnet50"])
    moco_v2.to(device)
    print("model set")

    ###### set optimizer, criterion ######
    optimizer = torch.optim.Adam(params = moco_v2.parameters(),
                                 lr = 1e-3,
                                 # need to tune hyperparams
                                 )
    criterion = nn.CrossEntropyLoss().to(device)
    print("optimizer are criterion are set")

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(moco_v2)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    loss_hist = AverageMeter()

    for e in range(opts.epochs):
        for batch_idx, (img_q, img_k) in enumerate(moco_trainloader):
            """
            @param img_q, img_k : (bs,3,224,224) 
            """
            img_q, img_k = img_q.to(device), img_k.to(device)
            # output : (bs, K+1); labels : (bs,)
            output, labels = moco_v2(img_q, img_k)
            output, labels = output.to(device), labels.to(device)

            loss = criterion(output, labels)
            loss_hist.update(val = loss.item(), n = opts.batch_size)

            loss.backward()
            optimizer.step()

            if batch_idx % opts.print_every == opts.print_every  - 1 :

                print("Train Epoch:{}, [{}/{}] Loss:{:.4f}/[avg: {:.4f}]".format(e,\
                                                                         batch_idx*opts.batch_size, \
                                                                         len(moco_trainloader.dataset),\
                                                                         loss_hist.val, loss_hist.avg))
if __name__ == "__main__":
    main()






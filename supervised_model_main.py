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

from moco_dataloader import SupervisedImageLoader
from moco_models import Resnet50
from utils import AverageMeter, bind_nsml, split_ids

import argparse

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Pretraining supervised encoder for EXPERIMENT1')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', default=128, type=int, help='BS')

parser.add_argument('--lr', default=.03, type=int, help='learning rate')
parser.add_argument('--sgd_momentum', default=.9, type=int, help='sgd momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay',dest='weight_decay')

parser.add_argument('--print_every', default=10, type=int, help='')

parser.add_argument('--name',default='Resnet', type=str, help='output model name')
parser.add_argument('--save_epoch', type=int, default=10, help='saving epoch interval')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
######################################################################

def main():
    print("torch version : ", torch.__version__)
    global opts
    opts = parser.parse_args()

    train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
    print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))

    res_trainloader = SupervisedImageLoader(DATASET_PATH, 'train', np.union1d(train_ids, val_ids),
                                       # simCLR style transform
                                       transform=transforms.Compose([
                                           transforms.Resize(opts.imResize),
                                           transforms.RandomResizedCrop(opts.imsize),
                                           # transforms.RandomApply([
                                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),  # not strengthened
                                                # ], p=0.8),
                                           transforms.RandomGrayscale(p=0.2),
                                           # gaussian blur should be added
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))
    print("len(res_trainloader) :", len(res_trainloader), )
    res_trainloader = torch.utils.data.DataLoader(res_trainloader, batch_size=opts.batch_size, shuffle=True,
                                                   pin_memory=True, drop_last=True) # num_workers = 4

    res_valloader = SupervisedImageLoader(DATASET_PATH, 'val', val_ids,
                                     # simCLR style transform
                                     transform=transforms.Compose([
                                           transforms.Resize(opts.imResize),
                                           transforms.RandomResizedCrop(opts.imsize),
                                           # transforms.RandomApply([
                                            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),  # not strengthened
                                                # ], p=0.8),
                                           transforms.RandomGrayscale(p=0.2),
                                           # gaussian blur should be added
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]))


    print("len(res_valloader) :", len(res_valloader), )
    res_valloader = torch.utils.data.DataLoader(res_valloader, batch_size=opts.batch_size, shuffle=False,
                                                   pin_memory=True, drop_last=False) # num_workers = 4
    print('train_loaders done')

    ###### set device, model ######
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {d}".format(d = device))

    ###### set model ######
    resnet50 = Resnet50(base_encoder=models.__dict__["resnet50"],)
    resnet50.to(device)
    print("model set")

    ###### set optimizer, criterion ######
    optimizer = torch.optim.SGD(resnet50.parameters(), opts.lr,
                                momentum=opts.sgd_momentum,
                                weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)
    print("optimizer are criterion are set")

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(resnet50)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    i = 0
    best_loss = float("inf")

    for e in range(opts.epochs):
        loss_hist = AverageMeter()
        for batch_idx, (imgs, labels) in enumerate(res_trainloader):
            """
            @param imgs : (bs,3,224,224) 
            """
            imgs, labels = imgs.to(device), labels.to(device)

            output = resnet50(imgs,)
            output = output.to(device)

            loss = criterion(output, labels)

            loss_hist.update(val = loss.item(), n = opts.batch_size)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()


            if batch_idx % opts.print_every == opts.print_every  - 1 :

                print("Train Epoch:{}, [{}/{}] Loss:{:.4f}/[avg: {:.4f}]".format(e+1,\
                                                                         batch_idx*opts.batch_size, \
                                                                         len(res_trainloader.dataset),\
                                                                         loss_hist.val, loss_hist.avg))
            if i % opts.print_every == opts.print_every - 1:
                nsml.report(step=i, loss=loss_hist.val, loss_avg=loss_hist.avg)
            i += 1

        if loss_hist.val < best_loss :
            print("saving best checkpoint... ")
            if IS_ON_NSML :
                nsml.save(opts.name + '_best')
            else :
                torch.save(resnet50.state_dict(), os.path.join('runs',opts.name+'_best'))
        # auto save on a basis of epoch interval
        if (e+1) % opts.save_epoch == 0:
            print("auto save ...")
            if IS_ON_NSML :
                nsml.save(opts.name + '_e{}'.format(e))
            else :
                torch.save(resnet50.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(e)))

        best_loss = min(loss_hist.val, best_loss)

if __name__ == "__main__":
    main()






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

from moco_dataloader import MoCoImageLoader
from moco_models import MoCoV2, MoCoClassifier, LinearProtocol
from utils import AverageMeter, bind_nsml, top_n_accuracy_score

import argparse

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Experiment1-Linear Protocol MoCo ')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', default=128, type=int, help='BS')

parser.add_argument('--lr', default=.03, type=int, help='learning rate')
parser.add_argument('--sgd_momentum', default=.9, type=int, help='sgd momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay', dest='weight_decay')

parser.add_argument('--print_every', default=10, type=int, help='')

parser.add_argument('--name', default='moco_linear', type=str, help='output model name')
parser.add_argument('--save_epoch', type=int, default=10, help='saving epoch interval')

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
    print("torch version : ", torch.__version__)
    global opts
    opts = parser.parse_args()

    train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
    print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))

    moco_trainloader = MoCoImageLoader(DATASET_PATH, 'val', train_ids,
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
                                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]), ]))
    print("len(moco_trainloader) :", len(moco_trainloader), )
    moco_trainloader = torch.utils.data.DataLoader(moco_trainloader, batch_size=opts.batch_size, shuffle=True,
                                                  num_workers = 4, pin_memory=True, drop_last=True)  # num_workers = 4

    moco_valloader = MoCoImageLoader(DATASET_PATH, 'val', val_ids,
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


    print("len(moco_valloader) :", len(moco_valloader), )
    moco_valloader = torch.utils.data.DataLoader(moco_valloader, batch_size=opts.batch_size, shuffle=False,
                                                   num_workers = 4, pin_memory=True, drop_last=False) # num_workers = 4

    print('train_loaders done')

    ###### set device, model ######
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {d}".format(d=device))

    ###### set model ######
    moco = MoCoV2(base_encoder=models.__dict__["resnet18"],)

    moco.to(device)
    print("moco loaded and saved")

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(moco)
        if opts.pause:
            nsml.paused(scope=locals())
    ######### pretrained moco #######################
    moco.train()
    nsml.load(checkpoint = 'MoCoV2_best', session = 'kaist_11/fashion_eval/7')
    nsml.save('saved')
    # exit()

    opts.finetuning = False # tune all layers
    if opts.finetuning :
        for param in moco.parameters():
            param.requires_grad = False
            assert len([param.size for param in moco.parameters() if param.requires_grad]) == 0


    linear_protocol = MoCoClassifier(moco, LinearProtocol())


    ### DO NOT MODIFY THIS BLOCK ### RE- BIND_NSML
    if IS_ON_NSML:
        bind_nsml(linear_protocol)
        if opts.pause:
            nsml.paused(scope=locals())

    ###### set optimizer, criterion ######
    linear_protocol_optimizer = torch.optim.SGD(linear_protocol.parameters(), opts.lr,
                                momentum=opts.sgd_momentum,
                                weight_decay=opts.weight_decay)
    linear_protocol.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    print("optimizer are criterion are set")


    ### linear_protocol train ########

    i = 0
    best_loss = float("inf")
    best_acc = -float("inf")

    for e in range(opts.epochs):
        loss_hist = AverageMeter()
        for batch_idx, (imgs, labels) in enumerate(moco_trainloader):
            linear_protocol.train()
            """
            @param imgs : (bs,3,224,224) 
            """
            imgs, labels = imgs.to(device), labels.to(device)

            output = linear_protocol(imgs,)
            output = output.to(device)

            loss = criterion(output, labels)

            loss_hist.update(val = loss.item(), n = opts.batch_size)

            linear_protocol_optimizer.zero_grad()
            loss.backward()

            linear_protocol_optimizer.step()


            if batch_idx % opts.print_every == opts.print_every  - 1 :

                print("Train Epoch:{}, [{}/{}] Loss:{:.4f}/[avg: {:.4f}]".format(e+1,\
                                                                         batch_idx*opts.batch_size, \
                                                                         len(moco_trainloader.dataset),\
                                                                         loss_hist.val, loss_hist.avg))
            if i % opts.print_every == opts.print_every - 1:
                nsml.report(step=i, loss=loss_hist.val, loss_avg=loss_hist.avg)

            i += 1


        print("start validation at the end of every epoch")
        acc_top1, acc_top5 = validate(opts, moco_valloader, linear_protocol, e, device)
        nsml.report(step=i, acc_top1=acc_top1, acc_top5=acc_top5)
        is_best = acc_top1 > best_acc
        best_acc = max(acc_top1, best_acc)
        if is_best :
            print("saving best_val checkpoint... ")
            if IS_ON_NSML:
                nsml.save(opts.name + '_best')
            else:
                torch.save(linear_protocol.state_dict(), os.path.join('runs', opts.name + '_best'))


        # auto save on a basis of epoch interval
        if (e+1) % opts.save_epoch == 0:
            print("auto save ...")
            if IS_ON_NSML :
                nsml.save(opts.name + '_e{}'.format(e))
            else :
                torch.save(linear_protocol.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(e)))

        best_loss = min(loss_hist.val, best_loss)

def validate(opts, res_valloader, linear_protocol, epoch, device):
    linear_protocol.eval()
    avg_top1, avg_top5  = 0., 0.
    cnt = 0
    with torch.no_grad():
        for batch_idx, (img, label) in enumerate(res_valloader):
            img, label = img.to(device), label.to(device)
            outputs = linear_protocol(img)

            acc_top1 = top_n_accuracy_score(label.cpu().numpy(), outputs.cpu().numpy(), n=1, normalize=True)*100
            acc_top5 = top_n_accuracy_score(label.cpu().numpy(), outputs.cpu().numpy(), n=5, normalize=True)*100
            avg_top1, avg_top5 = (avg_top1+acc_top1), (avg_top5+acc_top5)
            cnt += 1
    avg_top1, avg_top5 = avg_top1/cnt, avg_top5/cnt
    print("Validation Epoch:{}, Top1-acc:{:.4f}, Top5-acc:{:.4f}".format(epoch + 1, avg_top1, avg_top5))

    return avg_top1, avg_top5

if __name__ == "__main__":
    main()

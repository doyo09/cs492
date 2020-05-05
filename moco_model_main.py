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
from utils import AverageMeter, split_ids
from rand_aug.rand_augmentation import RandAugment

import argparse


def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx + 1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter

### NSML functions
def _moco_infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            MoCoImageLoader(root_path, 'test',
                            transform=transforms.Compose([
                                transforms.Resize(opts.imResize),
                                transforms.CenterCrop(opts.imsize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.batchsize, shuffle=False, num_workers = 4,pin_memory=True, drop_last=False)
        print('loaded {} validation images'.format(len(test_loader.dataset)))

    outputs = []
    # s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _moco_infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)
######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Pretraining MoCo')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', default=200, type=int, help='BS')

parser.add_argument('--lr', default=.03, type=int, help='learning rate')
parser.add_argument('--sgd_momentum', default=.9, type=int, help='sgd momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay', dest='weight_decay')

parser.add_argument('--print_every', default=10, type=int, help='')

parser.add_argument('--name', default='MoCoV2', type=str, help='output model name')
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

    moco_trainloader = MoCoImageLoader(DATASET_PATH, 'train', np.setdiff1d(unl_ids, val_ids),
                                       # unl_ids if you fully train moco
                                       # simCLR style transform
                                       transform=transforms.Compose([
                                           RandAugment(1,2),
                                           transforms.Resize([opts.imsize, opts.imsize]), # imResize
                                           # transforms.RandomResizedCrop(opts.imsize),
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
                                                   pin_memory=True, drop_last=True, num_workers = 4)  # num_workers = 4

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
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225]), ]))

    print("len(moco_valloader) :", len(moco_valloader), )
    moco_valloader = torch.utils.data.DataLoader(moco_valloader, batch_size=opts.batch_size, shuffle=False,
                                                 pin_memory=True, drop_last=False, num_workers = 4)
    print('train_loaders done')

    ###### set device, model ######
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {d}".format(d=device))

    ###### set model ######
    moco_v2 = MoCoV2(base_encoder=models.__dict__["resnet18"],)
    moco_v2.to(device)
    print("model set")

    ###### set optimizer, criterion ######
    optimizer = torch.optim.SGD(moco_v2.parameters(), opts.lr,
                                momentum=opts.sgd_momentum,
                                weight_decay=opts.weight_decay)
    criterion = nn.CrossEntropyLoss().to(device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[2,20,50],)
    print("optimizer are criterion are set")

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(moco_v2)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################
    # retrain
    moco_v2.train()
    nsml.load(checkpoint='MoCoV2_best', session='kaist_11/fashion_eval/264')
    nsml.save('saved')

    i = 0
    # j = 0
    best_loss = float("inf")

    for e in range(opts.epochs):
        loss_hist = AverageMeter()
        for batch_idx, (img_q, img_k) in enumerate(moco_trainloader):
            """
            @param img_q, img_k : (bs,3,224,224) 
            """

            img_q, img_k = img_q.to(device), img_k.to(device)
            # output : (bs, K+1); labels : (bs,)
            output, labels = moco_v2(img_q, img_k)
            output, labels = output.to(device), labels.to(device)

            loss = criterion(output, labels)
            loss_hist.update(val=loss.item(), n=opts.batch_size)

            optimizer.zero_grad()
            loss.backward()
            if e == 0:  # and batch_idx < 64:
                print("not updating until queue is full")
            else:
                optimizer.step()
            # print(loss.item())

            if batch_idx % opts.print_every == opts.print_every - 1:
                # print(moco_v2.queue, moco_v2.queue_pointer)
                print("Train Epoch:{}, [{}/{}] Loss:{:.4f}/[avg: {:.4f}]".format(e + 1, \
                                                                                 batch_idx * opts.batch_size, \
                                                                                 len(moco_trainloader.dataset), \
                                                                                 loss_hist.val, loss_hist.avg))
            if i % opts.print_every == 0:
                nsml.report(step=i, loss=loss_hist.val, loss_avg=loss_hist.avg)
            i += 1
        scheduler.step()
        # validation
        # model.eval()
        # avg_top1 = 0.0
        # avg_top5 = 0.0
        # nCnt = 0
        # with torch.no_grad():
        #     for batch_idx, (imgs, labels) in enumerate(res_valloader):
        #         imgs, labels = imgs.to(device), labels.to(device)
        #         nCnt += 1
        #         output = model(imgs)
        #
        #         acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1) * 100
        #         acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5) * 100
        #         avg_top1 += acc_top1
        #         avg_top5 += acc_top5
        #
        #     avg_top1 = float(avg_top1 / nCnt)
        #     avg_top5 = float(avg_top5 / nCnt)
        #     print('Test Epoch:{} Top1_acc_val:{:.2f}% Top5_acc_val:{:.2f}% '.format(epoch, avg_top1, avg_top5))
        #     ###### nsml report ######
        #     nsml.report(step=j, val_acc_top1=avg_top1, val_acc_top5=avg_top5, )
        # is_best = avg_top1 > best_acc
        # best_acc = max(avg_top1, best_acc)
        # j += 1

        if  loss_hist.val < best_loss :
            print('saving best checkpoint...')
            if IS_ON_NSML:
                nsml.save(opts.name + '_best')
            else:
                torch.save(moco_v2.state_dict(), os.path.join('runs', opts.name + '_best'))
        # auto save
        if (e + 1) % opts.save_epoch == 0:
            print('autosave...')
            if IS_ON_NSML:
                nsml.save(opts.name + '_e{}'.format(e))
            else:
                torch.save(moco_v2.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(e)))

        best_loss = min(loss_hist.val, best_loss)


if __name__ == "__main__":
    main()


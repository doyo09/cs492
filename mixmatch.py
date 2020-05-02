from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import numpy as np
import shutil
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision.models as models
from torchvision import datasets, transforms
from moco_dataloader import MixMatchImageLoader
from utils import AverageMeter, split_ids, get_tsa_threshold
from moco_models import MoCoV2, MoCoClassifier, Resnet50, ResnetClassifier, ClassifierBlock
from baseline.models import Res18_basic, Res50
from rand_aug.rand_augmentation import RandAugment

# from pytorch_metric_learning import miners
# from pytorch_metric_learning import losses as lossfunc
import glob

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

NUM_CLASSES = 265
if not IS_ON_NSML:
    DATASET_PATH = 'fashion_demo'


def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter


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

def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)


def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets


def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]


class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, final_epoch,
                 step, total_steps, # tsa related params
                 ood_mask_pct = .3, # ood related params : need to tune between .3 and .6
                 is_tsa = True, is_ood = True):
        # sup_loss
        if is_tsa :
            probs_sup = torch.softmax(outputs_x, dim=1) # (N,C)
            probs_sup = torch.sum(probs_sup * targets_x, dim = 1) #(N,C) -> (N,)
            tsa_threshold = get_tsa_threshold(curr_step=step, total_steps=total_steps, schedule="linear") #"exp", "log"
            larger_than_threshold = probs_sup > tsa_threshold
            loss_mask = torch.ones(targets_x.size(0), dtype=torch.float32).cuda() * (1-larger_than_threshold.type(torch.float32)).cuda()
            Lx = -torch.sum(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1) * loss_mask)
            Lx = Lx/ torch.sum(loss_mask,)
            # print("tsa_threshold is ", tsa_threshold.item(),
            #       "we mask ", targets_x.size(0) - torch.sum(loss_mask).item(), "out of ", targets_x.size(0),
            #       "Lx : ", Lx.item())
        else :
            Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

        # unlabel_loss
        # Lu = loss_l2u, logits_y = outputs_u, labels_y = targets_u
        probs_u = torch.softmax(outputs_u, dim=1) #(N,C)

        if is_ood:
            largest_probs, _ = torch.max(probs_u, dim = 1) # largest_prob : (N,)
            sorted_probs, _ = torch.sort(largest_probs, descending = False) #ascending probs : (N,)
            sort_index =  int(ood_mask_pct * sorted_probs.size(0)) # num of indice we need to mask
            ood_moving_threshold =  sorted_probs[sort_index]
            ood_mask = (largest_probs > ood_moving_threshold).type(torch.float32).cuda() # (N,) if false, mask them

            mse_loss = (probs_u - targets_u) ** 2 # (N,C)
            mse_loss = mse_loss * (ood_mask.unsqueeze(1).expand(mse_loss.size(0), mse_loss.size(1))) # (N,C)
            Lu = torch.sum(mse_loss) / (torch.sum(ood_mask,) * NUM_CLASSES)
            # print("sort_index : ", sort_index, ", ood_moving_thrh : ", ood_moving_threshold.item(),
            #       ", we mask ", targets_u.size(0) - torch.sum(ood_mask).item(), "out of ", targets_u.size(0),
            #       ", Lu : ", Lu.item())

        else : # not using ood
            Lu = torch.mean((probs_u - targets_u) ** 2)

        return Lx, Lu, opts.lambda_u * linear_rampup(epoch, final_epoch)



### NSML functions
def _moco_infer(model, root_path, test_loader=None):
    opts = parser.parse_args()
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            MixMatchImageLoader(root_path, 'test',
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
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')

# basic settings
parser.add_argument('--name', default='Res18baseMM', type=str, help='output model name')
parser.add_argument('--gpu_ids', default='0', type=str, help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=200, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=30, help='saving epoch interval')

# hyper-parameters for mix-match
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')


parser.add_argument('--use_sup_pretrained', action='store_true', help = 'use supervisely pretrained')
parser.add_argument('--use_moco', action='store_true', help = 'use moco')

parser.add_argument('--release', action='store_true', help = 'unfreeze more layers to update')
parser.add_argument('--finetuning', action='store_true', help = 'use finetuning with freezing the rest layers')
parser.add_argument('--retrain', action='store_true', help = 'retrain')

parser.add_argument('--use_tsa', action='store_true', help = 'tsa')
parser.add_argument('--use_ood', action='store_true', help = 'ood')

parser.add_argument('--use_randaug', action='store_true', help = 'rand augmentation')
parser.add_argument('--N', type=int,default=1, help = 'num of augmentations')
parser.add_argument('--M', type=int, default=2, help = 'magnitude')


################################

def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0

    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        device = torch.device('cuda')
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    # Set model
    if opts.use_sup_pretrained :
        resnet = Resnet50(base_encoder=models.__dict__["resnet50"], )
        resnet.to(device)
        print("resnet loaded and saved")
        ### DO NOT MODIFY THIS BLOCK ###
        if IS_ON_NSML:
            bind_nsml(resnet)
        ######### pretrained resnet load ######################
        resnet.train()
        nsml.load(checkpoint='Resnet_best', session='kaist_11/fashion_eval/19')
        # nsml.load(checkpoint='resnet_linear_e89', session='kaist_11/fashion_eval/31') # pretrained Classifier
        nsml.save('saved')
        if opts.finetuning:
            for param in resnet.parameters():
                param.requires_grad = False
                # print([param.size for param in resnet.parameters() if param.requires_grad])
            assert len([param.size for param in resnet.parameters() if param.requires_grad]) == 0

        model = ResnetClassifier(resnet, ClassifierBlock(), pretrained=True, init=True)
        del resnet
    elif opts.use_moco :
        moco = MoCoV2(base_encoder=models.__dict__["resnet50"],)
        moco.to(device)
        print("moco-pretrained loaded and saved")
        ### DO NOT MODIFY THIS BLOCK ###
        if IS_ON_NSML:
            bind_nsml(moco)
        ######### pretrained moco load #######################
        moco.train()
        nsml.load(checkpoint='MoCoV2_best', session='kaist_11/fashion_eval/7')
        nsml.save('saved')
        if opts.finetuning:
            for param in moco.parameters():
                param.requires_grad = False
            assert len([param.size for param in moco.parameters() if param.requires_grad]) == 0

        model = MoCoClassifier(moco, ClassifierBlock(), use_bn=True, pretrained=True, init=True)
        del moco
    else : # baseline
        model = Res18_basic(NUM_CLASSES)

        # model = Res50(NUM_CLASSES)




    model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if use_gpu:
        model.cuda()

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    if opts.retrain: # start training in the middle
        # for sup-pretrained
        if opts.use_sup_pretrained :
            nsml.load(checkpoint='Res18baseMM_best', session='kaist_11/fashion_eval/150') #150+Res18baseMM_best : trained for 200 epochs; 154+Res18baseMM_e49: +release_layers
            nsml.save('Res18baseMM_e200')
        # for moco-pretrained
        elif opts.use_moco :
            nsml.load(checkpoint='Res18baseMM_e49', session='kaist_11/fashion_eval/154')
            nsml.save('Res18baseMM_e49')
        else :
            nsml.load(checkpoint='Res18baseMM_best', session='kaist_11/fashion_eval/145')
            nsml.save('baseline_best')

    ########### when you train pretrained model ###########
    # nsml.load(checkpoint='Res18baseMM_best', session='kaist_11/fashion_eval/118')
    # nsml.save('saved')
    if IS_ON_NSML and opts.pause:
        nsml.paused(scope=locals())

    if opts.mode == 'train':
        model.train()
        # Set dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)
        print(
            'found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
        train_loader = torch.utils.data.DataLoader(
            MixMatchImageLoader(DATASET_PATH, 'train', train_ids,
                              transform=transforms.Compose([
                                  RandAugment(opts.N, opts.M) if opts.use_randaug else lambda x: x,
                                  transforms.Resize([opts.imsize, opts.imsize]), #opts.imResize
                                  # transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ColorJitter(brightness= 0.4,contrast=0.4,saturation=0.4, hue= 0.4),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('train_loader done')

        unlabel_loader = torch.utils.data.DataLoader(
            MixMatchImageLoader(DATASET_PATH, 'unlabel', unl_ids,
                              transform=transforms.Compose([
                                  RandAugment(opts.N, opts.M) if opts.use_randaug else lambda x: x,
                                  transforms.Resize([opts.imsize, opts.imsize]), #opts.imResize
                                  # transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ColorJitter(brightness= 0.4,contrast=0.4,saturation=0.4, hue= 0.4),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('unlabel_loader done')

        validation_loader = torch.utils.data.DataLoader(
            MixMatchImageLoader(DATASET_PATH, 'val', val_ids,
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.CenterCrop(opts.imsize),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print('validation_loader done')

        # Set optimizer
        optimizer = optim.Adam(model.parameters(), lr=opts.lr)

        # INSTANTIATE LOSS CLASS
        train_criterion = SemiLoss()

        # INSTANTIATE STEP LEARNING SCHEDULER CLASS
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 120, 150, 180], gamma=0.1, )

        # Train and Validation
        train_i = 0 # for nsml.report
        val_i = 0
        global train_i
        global val_i

        best_acc = -1

        if opts.finetuning and opts.release:
            release_schedule = np.linspace(1, opts.epochs, 5)[1:]

        for epoch in range(opts.start_epoch, opts.epochs + 1):
            ## release layers to update
            if opts.finetuning and opts.release:
                if list(release_schedule) :
                    if epoch > release_schedule[0]:
                        cnt = 0
                        for param in reversed(list(model.parameters())):
                            if not param.requires_grad:
                                param.requires_grad = True
                                cnt += 1
                                if cnt == 6: break
                            else:
                                continue
                        del cnt
                        release_schedule = release_schedule[1:]

            print('start training')
            loss, train_acc_top1, train_acc_top5, train_i  = train(opts, train_loader, unlabel_loader, model, train_criterion, optimizer, epoch, use_gpu, train_i)
            scheduler.step()

            print('start validation')
            acc_top1, acc_top5, val_i = validation(opts, validation_loader, model, epoch, use_gpu, val_i)
            is_best = acc_top1 > best_acc
            best_acc = max(acc_top1, best_acc)

            if is_best:
                print('saving best checkpoint...')
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best'))
            if (epoch + 1) % opts.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_e{}'.format(epoch)))


def train(opts, train_loader, unlabel_loader, model, criterion, optimizer, epoch, use_gpu, train_i):
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()
    weight_scale = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    avg_loss = 0.0
    avg_top1 = 0.0
    avg_top5 = 0.0

    model.train()

    nCnt = 0
    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)

    for batch_idx in range(len(train_loader)):
        try:
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        except:
            labeled_train_iter = iter(train_loader)
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        try:
            data = unlabeled_train_iter.next()
            inputs_u1, inputs_u2 = data
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            data = unlabeled_train_iter.next()
            inputs_u1, inputs_u2 = data

        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        classno = NUM_CLASSES
        targets_org = targets_x
        targets_x = torch.zeros(batch_size, classno).scatter_(1, targets_x.view(-1, 1), 1)

        if use_gpu:
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()
        inputs_x, targets_x = Variable(inputs_x), Variable(targets_x)
        inputs_u1, inputs_u2 = Variable(inputs_u1), Variable(inputs_u2)

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            embed_u1, pred_u1 = model(inputs_u1)
            embed_u2, pred_u2 = model(inputs_u2)
            pred_u_all = (torch.softmax(pred_u1, dim=1) + torch.softmax(pred_u2, dim=1)) / 2
            pt = pred_u_all ** (1 / opts.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            targets_u = targets_u.detach()

        # mixup
        all_inputs = torch.cat([inputs_x, inputs_u1, inputs_u2], dim=0)
        all_targets = torch.cat([targets_x, targets_u, targets_u], dim=0)

        lamda = np.random.beta(opts.alpha, opts.alpha)
        lamda = max(lamda, 1 - lamda)
        newidx = torch.randperm(all_inputs.size(0))
        input_a, input_b = all_inputs, all_inputs[newidx]
        target_a, target_b = all_targets, all_targets[newidx]

        mixed_input = lamda * input_a + (1 - lamda) * input_b
        mixed_target = lamda * target_a + (1 - lamda) * target_b

        # interleave labeled and unlabed samples between batches to get correct batchnorm calculation
        mixed_input = list(torch.split(mixed_input, batch_size))
        mixed_input = interleave(mixed_input, batch_size)

        optimizer.zero_grad()

        fea, logits_temp = model(mixed_input[0])
        logits = [logits_temp]
        for newinput in mixed_input[1:]:
            fea, logits_temp = model(newinput)
            logits.append(logits_temp)

        # put interleaved samples back
        logits = interleave(logits, batch_size)
        logits_x = logits[0]
        logits_u = torch.cat(logits[1:], dim=0)

        loss_x, loss_un, weigts_mixing = criterion(logits_x, mixed_target[:batch_size], logits_u,
                                                   mixed_target[batch_size:], epoch + batch_idx / len(train_loader),
                                                   opts.epochs,
                                                   step= int((epoch + batch_idx / len(train_loader)) * (len(train_loader.dataset)/batch_size)),
                                                   total_steps= int(opts.epochs *(len(train_loader.dataset)/batch_size)),
                                                   ood_mask_pct=.3, # ood related params : need to tune between .3 and .6
                                                   is_tsa = opts.use_tsa,
                                                   is_ood=opts.use_ood
                                                   )
        loss = loss_x + weigts_mixing * loss_un

        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(loss_x.item(), inputs_x.size(0))
        losses_un.update(loss_un.item(), inputs_x.size(0))
        weight_scale.update(weigts_mixing, inputs_x.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # compute guessed labels of unlabel samples
            embed_x, pred_x1 = model(inputs_x)

        acc_top1b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1) * 100
        acc_top5b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5) * 100
        acc_top1.update(torch.as_tensor(acc_top1b), inputs_x.size(0))
        acc_top5.update(torch.as_tensor(acc_top5b), inputs_x.size(0))

        avg_loss += loss.item()
        avg_top1 += acc_top1b
        avg_top5 += acc_top5b

        if batch_idx % opts.log_interval == 0:
            print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) Top-1:{:.2f}%({:.2f}%) Top-5:{:.2f}%({:.2f}%) '.format(
                epoch, batch_idx * inputs_x.size(0), len(train_loader.dataset), losses.val, losses.avg, acc_top1.val,
                acc_top1.avg, acc_top5.val, acc_top5.avg))

        nCnt += 1

        ###### nsml report ######
        nsml.report(step=train_i, train_loss=loss.item(), train_acc_top1=acc_top1b, train_acc_top5=acc_top5b,)
        train_i += 1

    avg_loss = float(avg_loss / nCnt)
    avg_top1 = float(avg_top1 / nCnt)
    avg_top5 = float(avg_top5 / nCnt)

    return avg_loss, avg_top1, avg_top5, train_i


def validation(opts, validation_loader, model, epoch, use_gpu, val_i):
    model.eval()
    avg_top1 = 0.0
    avg_top5 = 0.0
    nCnt = 0
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu:
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            nCnt += 1
            embed_fea, preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1) * 100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5) * 100
            avg_top1 += acc_top1
            avg_top5 += acc_top5

        avg_top1 = float(avg_top1 / nCnt)
        avg_top5 = float(avg_top5 / nCnt)
        print('Test Epoch:{} Top1_acc_val:{:.2f}% Top5_acc_val:{:.2f}% '.format(epoch, avg_top1, avg_top5))
        ###### nsml report ######
        nsml.report(step=val_i, val_acc_top1=avg_top1, val_acc_top5=avg_top5, )
        val_i += 1

    return avg_top1, avg_top5, val_i


if __name__ == '__main__':
    main()


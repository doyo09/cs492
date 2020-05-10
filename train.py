from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

import random
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.autograd import Variable

from utils import bind_nsml, split_ids, get_tsa_threshold, top_n_accuracy_score, interleave, linear_rampup
from utils import adjust_cosine_learning_rate, AverageMeter
from models import get_model_sup, get_model_moco, EMA
from baseline.models import Res18_basic, Res50
from dataloader import get_loader_hardmixmatch, get_loader_fixmatch, get_loader_baseline, get_loader_randaug

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Team 11 fashion_eval')

# basic settings
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
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
parser.add_argument('--sgd', action='store_true', default=False, help='use sgd scheduler')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=30, help='saving epoch interval')

# hyper-parameters for mix-match
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=75, type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--split_train_ratio', default=0.2, type=float)

# choose model, default: res50, if you use supervised model or moco, specify session and checkpoint
parser.add_argument('--model_sup', action='store_true', default=False, help='use supervisely pretrained')
parser.add_argument('--model_moco', action='store_true', default=False, help='use moco')
parser.add_argument('--model_session', type=str, default='', help='session name for pretrained model')
parser.add_argument('--model_checkpoint', type=str, default='', help='checkpoint for pretrained model')
parser.add_argument('--model_res18', action='store_true', default=False, help='use res18')

parser.add_argument('--release', action='store_true', help='unfreeze more layers to update')
parser.add_argument('--finetuning', action='store_true', help='use finetuning with freezing the rest layers')
parser.add_argument('--retrain', action='store_true', help='retrain')

# RandAug
parser.add_argument('--randaug', action='store_true', help='rand augmentation')
parser.add_argument('--N', type=int, default=1, help='num of augmentations')
parser.add_argument('--M', type=int, default=2, help='magnitude')

# other regularization techniques
parser.add_argument('--tsa', action='store_true', help='tsa')
parser.add_argument('--ood', action='store_true', help='ood')
parser.add_argument('--ema', action='store_true', default=False, help='ema')
parser.add_argument('--ema-decay', default=0.999, type=float, help='EMA decay rate')

# hard mixmatch (randaug automatically enabled)
parser.add_argument('--hardmixmatch', action='store_true', default=False, help='use hard mixmatch')
parser.add_argument('--hardmixmatch_threshold', type=float, default=0.5, help='threshold')
parser.add_argument('--hardmixmatch_celoss', action='store_true', default=False, help='use ce loss')

# fixmatch (randaug automatically enabled)
parser.add_argument('--fixmatch', action='store_true', default=False, help='use fixmatch')
parser.add_argument('--fixmatch_threshold', type=float, default=0.95, help='threshold')
parser.add_argument('--fixmatch_co', type=float, default=1, help='fixmatch coefficient')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

NUM_CLASSES = 265
if not IS_ON_NSML:
    DATASET_PATH = 'fashion_demo'

def main():
    global opts
    opts = parser.parse_args()

    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        device = torch.device('cuda')
        torch.cuda.manual_seed_all(seed)
    else:
        raise Exception("Using CPU is not supported")

    # Set model
    if opts.model_sup:
        model = get_model_sup(opts)
        print('model: supervised pretrained')
    elif opts.model_moco:
        model = get_model_moco(opts)
        print('model: moco')
    elif opts.model_res18:
        model = Res18_basic(NUM_CLASSES)
        print('model: Res18')
    else:
        model = Res50(NUM_CLASSES)
        print('model: Res50')
    model.eval()

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if use_gpu:
        model.cuda()


    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model, opts)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################

    if opts.mode == 'train':
        # for nsml.report
        train_i = 0
        val_i = 0
        best_acc = -1

        model.train()

        # setup dataloader
        train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), opts.split_train_ratio)

        if opts.fixmatch:
            train_loader, unlabel_loader, validation_loader = get_loader_fixmatch(DATASET_PATH, train_ids, unl_ids, val_ids, opts)
            print('using fixmatch')
        elif opts.randaug:
            train_loader, unlabel_loader, validation_loader = get_loader_randaug(DATASET_PATH, train_ids, unl_ids, val_ids, opts)
            print('using randaug')
        elif opts.hardmixmatch:
            train_loader, unlabel_loader, validation_loader = get_loader_hardmixmatch(DATASET_PATH, train_ids, unl_ids, val_ids, opts)
            print('using hard mixmatch')
        else:
            train_loader, unlabel_loader, validation_loader = get_loader_baseline(DATASET_PATH, train_ids, unl_ids, val_ids, opts)
            print('using baseline')

        # Set optimizer
        if opts.sgd:
            optimizer = optim.SGD(model.parameters(), lr=opts.lr, momentum=0.9, nesterov=True)
        else:
            optimizer = optim.Adam(model.parameters(), lr=opts.lr)

        # Instantiate step learning scheduler class
        if opts.hardmixmatch:
            scheduler = adjust_cosine_learning_rate(opts, optimizer, num_training_steps=int(opts.epochs *(len(train_loader.dataset)/opts.batchsize)))
        else:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 80, 120, 150, 180], gamma=0.1)

        ema_model = EMA(opts, model, opts.ema_decay, device) if opts.ema else None

        # Train and Validation

        if opts.finetuning and opts.release:
            release_schedule = np.linspace(1, opts.epochs, 5)[1:]

        for epoch in range(opts.start_epoch, opts.epochs + 1):
            ## release layers to update
            if opts.finetuning and opts.release and list(release_schedule) and epoch > release_schedule[0]:
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
            loss, train_acc_top1, train_acc_top5, train_i = train(opts, train_loader, unlabel_loader, model, optimizer, epoch, use_gpu, train_i, ema_model, scheduler)
            if opts.fixmatch:
                scheduler.step()

            print('start validation')
            if opts.ema:
                acc_top1, acc_top5, val_i = validation(opts, validation_loader, ema_model.ema, epoch, use_gpu, val_i)
            else:
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



def train(opts, train_loader, unlabel_loader, model, optimizer, epoch, use_gpu, train_i, ema_model, scheduler):
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()
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
        inputs_u1_str = None
        inputs_u2_str = None

        try:
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        except:
            labeled_train_iter = iter(train_loader)
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        try:
            data = unlabeled_train_iter.next()
            if opts.hardmixmatch:
                inputs_u1, inputs_u2, inputs_u1_str, inputs_u2_str = data
            else:
                inputs_u1, inputs_u2 = data
        except:
            unlabeled_train_iter = iter(unlabel_loader)
            data = unlabeled_train_iter.next()
            if opts.hardmixmatch:
                inputs_u1, inputs_u2, inputs_u1_str, inputs_u2_str = data
            else:
                inputs_u1, inputs_u2 = data

        batch_size = inputs_x.size(0)
        targets_org = targets_x
        # Transform label to one-hot
        targets_x_hot = torch.zeros(batch_size, NUM_CLASSES).scatter_(1, targets_x.view(-1, 1), 1)

        if use_gpu:
            inputs_x, targets_x, targets_x_hot = inputs_x.cuda(), targets_x.cuda(), targets_x_hot.cuda()
            inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()
            if opts.hardmixmatch:
                inputs_u1_str, inputs_u2_str = inputs_u1_str.cuda(), inputs_u2_str.cuda()

        optimizer.zero_grad()

        if opts.fixmatch:
            loss, loss_x, loss_un = fixmatch_loss(opts, model, batch_size, inputs_x, inputs_u1, inputs_u2, targets_x)
        else:
            rampup_epoch = epoch + batch_idx / len(train_loader)
            step = int((epoch + batch_idx / len(train_loader)) * (len(train_loader.dataset)/batch_size))
            total_steps = int(opts.epochs * (len(train_loader.dataset)/batch_size))
            loss, loss_x, loss_un, cnt_ge_thrh = mixup_loss(opts, model, batch_size, rampup_epoch, step, total_steps,
                                                inputs_x, inputs_u1, inputs_u2, inputs_u1_str, inputs_u2_str, targets_x_hot)


        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(loss_x.item(), inputs_x.size(0))
        losses_un.update(loss_un.item(), inputs_x.size(0))

        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if opts.hardmixmatch : 
            scheduler.step()
        if opts.ema:
            ema_model.update(model)

        # compute guessed labels of unlabel samples
        with torch.no_grad():
            _, pred_x1 = model(inputs_x)

        acc_top1b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1) * 100
        acc_top5b = top_n_accuracy_score(targets_org.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5) * 100
        acc_top1.update(torch.as_tensor(acc_top1b), inputs_x.size(0))
        acc_top5.update(torch.as_tensor(acc_top5b), inputs_x.size(0))

        avg_loss += loss.item()
        avg_top1 += acc_top1b
        avg_top5 += acc_top5b

        if batch_idx % opts.log_interval == 0:
            if opts.hardmixmatch:
                print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) Top-1:{:.2f}%({:.2f}%) Top-5:{:.2f}%({:.2f}%) loss_x:{:.4f} / loss_un:{:.4f} cnt_ge_thrh:{:f}'.format(epoch, batch_idx * inputs_x.size(0), len(train_loader.dataset), losses.val, losses.avg, acc_top1.val, acc_top1.avg, acc_top5.val, acc_top5.avg, losses_x.val, losses_un.val, cnt_ge_thrh))
            else: 
                print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) Top-1:{:.2f}%({:.2f}%) Top-5:{:.2f}%({:.2f}%) loss_x:{:.4f} / loss_un:{:.4f}'.format(
                epoch, batch_idx * inputs_x.size(0), len(train_loader.dataset), losses.val, losses.avg, acc_top1.val,
                acc_top1.avg, acc_top5.val, acc_top5.avg, losses_x.val, losses_un.val))

        # nsml report
        nsml.report(step=train_i, train_loss=loss.item(), train_acc_top1=acc_top1b, train_acc_top5=acc_top5b,)
        train_i += 1
        nCnt += 1

    avg_loss = float(avg_loss / nCnt)
    avg_top1 = float(avg_top1 / nCnt)
    avg_top5 = float(avg_top5 / nCnt)

    return avg_loss, avg_top1, avg_top5, train_i


def fixmatch_loss(opts, model, batch_size, inputs_x, inputs_u1, inputs_u2, targets_x):
    inputs = torch.cat((inputs_x, inputs_u1, inputs_u2)).cuda()
    _, logits = model(inputs)
    logits_x = logits[:batch_size]
    logits_u_w, logits_u_s = logits[batch_size:].chunk(2)
    del logits

    loss_x = F.cross_entropy(logits_x, targets_x, reduction='mean')

    pseudo_label = torch.softmax(logits_u_w.detach_(), dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(opts.fixmatch_threshold).float()

    loss_un = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()

    loss = loss_x + opts.fixmatch_co * loss_un

    return loss, loss_x, loss_un


def mixup_loss(opts, model, batch_size, epoch, step, total_steps,
                inputs_x, inputs_u1, inputs_u2, inputs_u1_str, inputs_u2_str, targets_x):
    with torch.no_grad():
        # compute guessed labels of unlabel samples
        _, pred_u1 = model(inputs_u1)
        _, pred_u2 = model(inputs_u2)
        pred_u_all = (torch.softmax(pred_u1, dim=1) + torch.softmax(pred_u2, dim=1)) / 2
        pseudo_label_idx = None
        # sharpening or pseudo-label
        if opts.hardmixmatch and epoch > 20:
            pred_u_max_idx = torch.max(pred_u_all, 1)[1] # (B)
            pseudo_label_idx = pred_u_all[torch.arange(pred_u_all.size(0)), pred_u_max_idx] > opts.hardmixmatch_threshold
            pt = pred_u_all ** (1 / opts.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
            # 하나라도 true면 실행
            if pseudo_label_idx.any():
                _, indice = torch.max(pt[pseudo_label_idx], dim=1)
                targets_u[pseudo_label_idx] = torch.zeros(targets_u[pseudo_label_idx].size()).cuda().scatter_(1, indice.view(-1, 1), 1)
        # sharpening
        else:
            pt = pred_u_all ** (1 / opts.T)
            targets_u = pt / pt.sum(dim=1, keepdim=True)
        targets_u = targets_u.detach()

    if opts.hardmixmatch:
        all_inputs = torch.cat([inputs_x, inputs_u1_str, inputs_u2_str], dim=0)
    else:
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

    _, logits_temp = model(mixed_input[0])
    logits = [logits_temp]
    for newinput in mixed_input[1:]:
        _, logits_temp = model(newinput)
        logits.append(logits_temp)

    # put interleaved samples back
    logits = interleave(logits, batch_size)
    logits_x = logits[0]
    logits_u = torch.cat(logits[1:], dim=0)

    loss_x, loss_un, weigts_mixing = mix_criterion(
        opts, logits_x, mixed_target[:batch_size], logits_u,
        mixed_target[batch_size:], epoch,
        step=step, total_steps=total_steps, # tsa related params
        ood_mask_pct=.3, # ood related params : need to tune between .3 and .6
    )
    loss = loss_x + weigts_mixing * loss_un

    return loss, loss_x, loss_un, torch.sum(pseudo_label_idx).item() if pseudo_label_idx is not None else 0.


def mix_criterion(opts, outputs_x, targets_x, outputs_u, targets_u, epoch, step, total_steps, ood_mask_pct = .3):
    # sup_loss
    if opts.tsa:
        probs_sup = torch.softmax(outputs_x, dim=1) # (N,C)
        probs_sup = torch.sum(probs_sup * targets_x, dim = 1) #(N,C) -> (N,)
        if opts.hardmixmatch:
            tsa_threshold = get_tsa_threshold(curr_step=step, total_steps=total_steps, schedule="log")
        else:
            tsa_threshold = get_tsa_threshold(curr_step=step, total_steps=total_steps, schedule="linear")
        larger_than_threshold = probs_sup > tsa_threshold
        loss_mask = torch.ones(targets_x.size(0), dtype=torch.float32).cuda() * (1-larger_than_threshold.type(torch.float32)).cuda()
        Lx = -torch.sum(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1) * loss_mask)
        Lx = Lx/ torch.sum(loss_mask,)
    else :
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))

    # unlabel_loss
    probs_u = torch.softmax(outputs_u, dim=1) #(N,C)

    if opts.ood:
        largest_probs, _ = torch.max(probs_u, dim=1) # largest_prob : (N,)
        sorted_probs, _ = torch.sort(largest_probs, descending=False) #ascending probs : (N,)
        sort_index = int(ood_mask_pct * sorted_probs.size(0)) # num of indice we need to mask
        ood_moving_threshold = sorted_probs[sort_index]
        ood_mask = (largest_probs > ood_moving_threshold).type(torch.float32).cuda() # (N,) if false, mask them

        if opts.hardmixmatch_celoss:
            ce_loss = -torch.sum(torch.sum(torch.log(probs_u) * targets_u, dim=1) * ood_mask) #(N,C) -> (N,) -> ()
            Lu = ce_loss / torch.sum(ood_mask,)
        else:
            mse_loss = (probs_u - targets_u) ** 2 # (N,C)
            mse_loss = mse_loss * (ood_mask.unsqueeze(1).expand(mse_loss.size(0), mse_loss.size(1))) # (N,C)
            Lu = torch.sum(mse_loss) / (torch.sum(ood_mask,) * NUM_CLASSES)

    else: # not using ood
        Lu = torch.mean((probs_u - targets_u) ** 2)

    return Lx, Lu, opts.lambda_u * linear_rampup(epoch, opts.epochs)



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

        # nsml report
        nsml.report(step=val_i, val_acc_top1=avg_top1, val_acc_top5=avg_top5, )
        val_i += 1

    return avg_top1, avg_top5, val_i



if __name__ == '__main__':
    main()

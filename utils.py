import os
import time
import math

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.optim.lr_scheduler import LambdaLR

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from dataloader import MixMatchImageLoader


### NSML functions
def _infer(model, root_path, opts, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            MixMatchImageLoader(root_path, 'test',
                            transform=transforms.Compose([
                                transforms.Resize(opts.imResize),
                                transforms.CenterCrop(opts.imsize),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print('loaded {} validation images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        _, probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model, opts, loader=None):
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
        return _infer(model, root_path, opts, loader)

    nsml.bind(save=save, load=load, infer=infer)

def load_model(model, opts):
    if IS_ON_NSML:
        bind_nsml(model, opts)
    nsml.load(checkpoint=opts.model_checkpoint, session=opts.model_session)
    nsml.save('saved')
    print('loaded model from : {} / {}'.format(opts.model_checkpoint, opts.model_session))


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


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


# TSA
def get_tsa_threshold(curr_step, total_steps, start=0, end=None, schedule="log-schdule", class_num=265):
    """
    :param curr_step:
    :param total_steps:
    :param start: starting step
    :param end:
    :param schedule: log or linear or exp
    :param class_num: 265
    :return: threshold
    """
    if end is None :
        end = total_steps
    # curr_step/total_steps
    frac_t_T = torch.tensor(curr_step/total_steps, dtype = torch.float32)
    if schedule.startswith("linear"):
        alpha_t = frac_t_T
    elif  schedule.startswith("exp"):
        scale = 5
        alpha_t = torch.exp((frac_t_T-1) * scale)
    elif schedule.startswith("log"):
        scale = 5
        alpha_t = 1- torch.exp(-frac_t_T*scale)
    else :
        raise ValueError("no schedule")
    threshold = alpha_t * (1 - 1/class_num) + 1/class_num

    return threshold


def adjust_cosine_learning_rate(opts, optimizer, num_training_steps, num_cycles=7./16., last_epoch=-1):
    """Sets the learning rate to the initial LR decayed by cosine function every step not epoch"""
    def _lambda_lr(current_step) :
        no_progress = float(current_step) / float(max(1, num_training_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lambda_lr, last_epoch)


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


def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)

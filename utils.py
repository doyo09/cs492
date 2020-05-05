from moco_dataloader import MoCoImageLoader
from baseline.ImageDataLoader import SimpleImageLoader
import nsml

import os
import torch
from torchvision import transforms


import numpy as np
import time

import torch.nn.functional as F


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
def get_tsa_threshold(curr_step, total_steps, start = 0, end = None,schedule = "log-schdule", class_num=265):
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


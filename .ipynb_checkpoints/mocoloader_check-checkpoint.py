import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch

from dataloader import MoCoImageLoader, SupervisedImageLoader

import argparse

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Pretraining MoCo')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

global opts
opts = parser.parse_args()

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
    cut = int(ratio*len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]

    return train_ids, val_ids, ids_u

    
train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.2)

print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))

# 비교를 위해 val_ids를 제외한다.
# 훈련시에는 모든 unl_ids 사용
moco_trainloader = MoCoImageLoader(DATASET_PATH, 'train', np.setdiff1d(unl_ids,val_ids), # unl_ids if you fully train moco
                                  transform=transforms.Compose([
                                      transforms.Resize(opts.imResize),
                                      transforms.RandomResizedCrop(opts.imsize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
print("len(moco_trainloader) :" , len(moco_trainloader),)

moco_valloader = MoCoImageLoader(DATASET_PATH, 'val', val_ids, 
                                  transform=transforms.Compose([
                                      transforms.Resize(opts.imResize),
                                      transforms.RandomResizedCrop(opts.imsize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
print("len(moco_valloader) :" , len(moco_valloader), )

sup_trainloader = SupervisedImageLoader(DATASET_PATH, 'train', train_ids, 
                                  transform=transforms.Compose([
                                      transforms.Resize(opts.imResize),
                                      transforms.RandomResizedCrop(opts.imsize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
print("len(sup_trainloader) :" , len(sup_trainloader), )

sup_valloader = SupervisedImageLoader(DATASET_PATH, 'val', val_ids, 
                                  transform=transforms.Compose([
                                      transforms.Resize(opts.imResize),
                                      transforms.RandomResizedCrop(opts.imsize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]))
print("len(sup_valloader) :" , len(sup_valloader))

print('train_loaders done')

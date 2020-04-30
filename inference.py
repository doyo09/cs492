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

from moco_dataloader import MoCoImageLoader, SupervisedImageLoader, MixMatchImageLoader
from moco_models import MoCoV2, MoCoClassifier, Resnet50, ResnetClassifier, ClassifierBlock
from baseline.models import Res18_basic, Res50

import argparse


### NSML functions
def _moco_infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            MixMatchImageLoader(root_path, 'test',
                            transform=transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=64, shuffle=False, num_workers = 4,pin_memory=True, drop_last=False)
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
    print("outputs", outputs, len(outputs))
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
parser = argparse.ArgumentParser(description='inference')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train (default: 200)')
parser.add_argument('--batch_size', default=128, type=int, help='BS')

parser.add_argument('--lr', default=.03, type=int, help='learning rate')
parser.add_argument('--sgd_momentum', default=.9, type=int, help='sgd momentum')
parser.add_argument('--weight-decay', default=1e-4, type=float, help='weight decay', dest='weight_decay')

parser.add_argument('--print_every', default=10, type=int, help='')

parser.add_argument('--name', default='resnet_linear', type=str, help='output model name')
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

    ###### set device, model ######
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using {d}".format(d=device))

    ###### set model ######
    resnet = Resnet50(base_encoder=models.__dict__["resnet50"],)
    model = ResnetClassifier(resnet, ClassifierBlock(), pretrained=True, init=True)

    model.to(device)
    del resnet

    ### DO NOT MODIFY THIS BLOCK ### RE- BIND_NSML
    if IS_ON_NSML:
        bind_nsml(model)
        if opts.pause:
            nsml.paused(scope=locals())

    ######### pretrained resnet #######################
    model.train()
    nsml.load(checkpoint = 'Res18baseMM_best', session = 'kaist_11/fashion_eval/195')
    nsml.save('savedmodel')
    # exit()


if __name__ == '__main__':
    main()

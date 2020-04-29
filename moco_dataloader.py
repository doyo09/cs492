from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch



def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        img_q = self.transform(inp)
        img_k = self.transform(inp)
        return img_q, img_k    
    
# all unlabeled data!
class MoCoImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader):
        """
        @param rootdir : DATASET_PATH from nsml ex)fashion_eval
        @param split : one of {"train", "val"}
        @ids : train id or val id
        @transform : transforms
        @loader : url to image 
        
        """
        if split in {"train", "val"}:
            self.impath = os.path.join(rootdir, 'train/train_data')
            # but we only need data; 
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        imids = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()
#                 print(instance_id, label)
                if int(label) == -1 and split != 'train':
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if int(instance_id) in imids:
                        continue
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        imids.append(int(instance_id))
                        if split == 'val':
                            imclasses.append(int(label))

        # transformations
        self.transform = transform 
        # generating imgs for query and imgs for key
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        if self.split == 'val':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        else:        
            img_q, img_k = self.TransformTwice(img)
            return img_q, img_k
        
    def __len__(self):
        return len(self.imnames)
    

# Build a supervied model to measure performance of MoCo's img representation 
# - only need labeled data
class SupervisedImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader):
        """
        
        @param rootdir : DATASET_PATH from nsml ex)fashion_eval
        @param split : one of {"train", "val"}; train should be all labeled data
        @ids : train id or val id
        @transform : transforms
        @loader : url to image
        
        """
        if split in {"train", "val"}:
            self.impath = os.path.join(rootdir, 'train/train_data')
            # but we only need data; 
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        imids = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                if (ids is None) or (int(instance_id) in ids):
                    if int(instance_id) in imids:
                        continue
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        imclasses.append(int(label))
                        imids.append(int(instance_id))

        self.transform = transform
#         self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses

    
    def __getitem__(self, index):
        
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        if self.transform is not None:
            img = self.transform(img)
            label = self.imclasses[index]
        return img, label
        
    def __len__(self):
        return len(self.imnames)


class MixMatchImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []

        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()
                if int(label) == -1 and (split != 'unlabel' and split != 'test'):
                    continue
                if int(label) != -1 and (split == 'unlabel' or split == 'test'):
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses

    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))

        if self.split == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.split != 'unlabel':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        else:
            img1, img2 = self.TransformTwice(img)
            return img1, img2

    def __len__(self):
        return len(self.imnames)

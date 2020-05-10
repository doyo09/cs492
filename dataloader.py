from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import torch

from randaug import RandAugment

def get_loader_randaug(DATASET_PATH, train_ids, unl_ids, val_ids, opts):
    label_transform = transforms.Compose([
        RandAugment(opts.N, opts.M) if opts.randaug else lambda x: x,
        transforms.Resize([opts.imsize, opts.imsize]), #opts.imResize
        transforms.RandomResizedCrop(opts.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    unlabel_transform = transforms.Compose([
        RandAugment(opts.N, opts.M) if opts.randaug else lambda x: x,
        transforms.Resize([opts.imsize, opts.imsize]), #opts.imResize
        transforms.RandomResizedCrop(opts.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.CenterCrop(opts.imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        MixMatchImageLoader(DATASET_PATH, 'train', train_ids, transform=label_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('train_loader done')

    unlabel_loader = torch.utils.data.DataLoader(
        MixMatchImageLoader(DATASET_PATH, 'unlabel', unl_ids, transform=unlabel_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('unlabel_loader done')

    validation_loader = torch.utils.data.DataLoader(
        MixMatchImageLoader(DATASET_PATH, 'val', val_ids, transform=val_transform),
        batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print('validation_loader done')

    return train_loader, unlabel_loader, validation_loader


def get_loader_hardmixmatch(DATASET_PATH, train_ids, unl_ids, val_ids, opts):
    label_transform = transforms.Compose([
        RandAugment(opts.N, opts.M) if opts.randaug else lambda x: x,
        transforms.Resize([opts.imsize, opts.imsize]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue= 0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    unlabel_transform = transforms.Compose([
        transforms.Resize([opts.imsize, opts.imsize]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_strong = transforms.Compose([
        RandAugment(opts.N, opts.M) if opts.randaug else lambda x: x,
        transforms.Resize([opts.imsize, opts.imsize]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.CenterCrop(opts.imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        HardMixMatchImageLoader(DATASET_PATH, 'train', train_ids, transform=label_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('train_loader done')

    unlabel_loader = torch.utils.data.DataLoader(
        HardMixMatchImageLoader(DATASET_PATH, 'unlabel', unl_ids, transform=unlabel_transform, transform_strong=transform_strong),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('unlabel_loader done')

    validation_loader = torch.utils.data.DataLoader(
        HardMixMatchImageLoader(DATASET_PATH, 'val', val_ids, transform=val_transform),
        batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print('validation_loader done')

    return train_loader, unlabel_loader, validation_loader


def get_loader_fixmatch(DATASET_PATH, train_ids, unl_ids, val_ids, opts):
    weak_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(opts.imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    label_transform = weak_transform

    unlabel_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(opts.imsize),
        RandAugment(opts.N, opts.M),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = weak_transform

    train_loader = torch.utils.data.DataLoader(
        FixMatchImageLoader(DATASET_PATH, 'train', train_ids, transform=label_transform, weak_transform=weak_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('train_loader done')

    unlabel_loader = torch.utils.data.DataLoader(
        FixMatchImageLoader(DATASET_PATH, 'unlabel', unl_ids, transform=unlabel_transform, weak_transform=weak_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('unlabel_loader done')

    validation_loader = torch.utils.data.DataLoader(
        FixMatchImageLoader(DATASET_PATH, 'val', val_ids, transform=val_transform, weak_transform=weak_transform),
        batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print('validation_loader done')

    return train_loader, unlabel_loader, validation_loader


def get_loader_baseline(DATASET_PATH, train_ids, unl_ids, val_ids, opts):
    train_transform = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.RandomResizedCrop(opts.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    unlabel_transform = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.RandomResizedCrop(opts.imsize),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    validation_transform = transforms.Compose([
        transforms.Resize(opts.imResize),
        transforms.CenterCrop(opts.imsize),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'train', train_ids, transform=train_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('train_loader done')

    unlabel_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'unlabel', unl_ids, transform=unlabel_transform),
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('unlabel_loader done')    

    validation_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'val', val_ids, transform=validation_transform),
        batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print('validation_loader done')

    return train_loader, unlabel_loader, validation_loader


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TransformTwice:
    def __init__(self, transform, weak_transform):
        self.transform = transform
        self.weak_transform = weak_transform
    def __call__(self, inp):
        if self.weak_transform != None:
            img_q = self.weak_transform(inp)
        else:
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
        self.TransformTwice = TransformTwice(transform, None)
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
        self.TransformTwice = TransformTwice(transform, None)
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


class FixMatchImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, weak_transform=None, loader=default_image_loader):
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
        self.TransformTwice = TransformTwice(transform, weak_transform)
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


class HardMixMatchImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, transform_strong=None, loader=default_image_loader):
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
        self.TransformTwice = TransformTwice(transform, None)
        self.TransformTwice_strong = TransformTwice(transform_strong, None) if transform_strong else None
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
        self.four_imgs = True if transform_strong else False


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

            if self.four_imgs:
                img1_s, img2_s =  self.TransformTwice_strong(img)
                return img1, img2, img1_s, img2_s
            return img1, img2

    def __len__(self):
        return len(self.imnames)


class SimpleImageLoader(torch.utils.data.Dataset):
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
        self.TransformTwice = TransformTwice(transform, None)
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

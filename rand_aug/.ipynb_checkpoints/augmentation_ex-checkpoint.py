
from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch
from rand_augmentation import RandAugment
import matplotlib.pyplot as plt

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class RandaugLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, ids=None, transform=None, loader=default_image_loader):
        self.impath = os.path.join(rootdir, 'train/train_data')
        meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []

        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.loader = loader
        self.imnames = imnames
        self.imclasses = imclasses

    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))

        if self.transform is not None:
            img_transformed = self.transform(img)
        return img, img_transformed

    def __len__(self):
        return len(self.imnames)
def main():
    train_loader = torch.utils.data.DataLoader(
        RandaugLoader(DATASET_PATH, list(range(21)),
                            transform=transforms.Compose([
                                RandAugment(2, 3) if opts.use_randaug else lambda x: x,
                                transforms.Resize(256),
                                transforms.RandomResizedCrop(224),
                                transforms.RandomHorizontalFlip(),
                                transforms.RandomVerticalFlip(),
                                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4),
                                #transforms.ToTensor(),
                            #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                            ])),
        batch_size=5, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('train_loader done')

    for i  in range(len(train_loader)): # 4
        for j in range(train_loader.batch_size): # 5
            img, img_transformed = next(train_loader)

            plt.subplot(len(train_loader), len(train_loader.batch_size)*2, i*len(train_loader.batch_size)*2+j*2+1)
            plt.imshow(img)

            plt.subplot(len(train_loader), len(train_loader.batch_size) * 2, i*len(train_loader.batch_size)*2+j*2+2)
            plt.imshow(img_transformed)
    plt.show()



if __name__ == '__main__':
    main()



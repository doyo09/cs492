from moco_dataloader import MoCoImageLoader
import nsml
from nsml import DATASET_PATH, IS_ON_NSML

import time

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

### NSML functions
def _moco_infer(model, root_path, val_loader=None):
    if val_loader is None:
        val_loader = torch.utils.data.DataLoader(
            MoCoImageLoader(root_path, 'val', val_ids,
                            transform=transforms.Compose([
                                transforms.Resize(opts.imResize),
                                transforms.RandomResizedCrop(opts.imsize),
                                # transforms.RandomApply([
                                #     transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
                                #      ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                # gaussian blur should be added
                                transforms.RandomHorizontalFlip(),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])),
            batch_size=opts.batch_size, shuffle=False, num_workers = 8,pin_memory=True, drop_last=False)
        print('loaded {} validation images'.format(len(val_loader.dataset)))

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



import os
import cv2
import numpy as np
import torch
import torch.utils.data as data
from torchvision.transforms.functional import (
    adjust_brightness, adjust_contrast,
    adjust_hue, adjust_saturation, normalize
)
from models.basicsr.utils import FileClient, get_root_logger, imfrombytes, img2tensor
from models.basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class MyCustomDataset(data.Dataset):
    def __init__(self, opt):
        super(MyCustomDataset, self).__init__()
        logger = get_root_logger()
        self.opt = opt


        self.gt_folder = opt['dataroot_gt']
        self.label_folder = opt.get('label_folder', None)


        self.paths = [os.path.join(self.gt_folder, f) for f in os.listdir(self.gt_folder) if f.endswith('.jpg')]


        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        assert self.gt_size >= self.in_size, 'Wrong setting.'

        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])

    def __getitem__(self, index):

        img_path = self.paths[index]
        img = imfrombytes(FileClient('disk', io_backend='disk').get(img_path), float32=True)


        if self.opt.get('use_hflip', False) and np.random.rand() > 0.5:
            img = cv2.flip(img, 1)


        label = None
        if self.label_folder:
            label_name = os.path.basename(img_path).replace('.jpg', '.txt')
            label_path = os.path.join(self.label_folder, label_name)
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    label = f.read().strip()


        img = cv2.resize(img, (self.gt_size, self.gt_size))
        img = img / 255.0


        img_tensor = img2tensor(img, bgr2rgb=True, float32=True)
        normalize(img_tensor, self.mean, self.std, inplace=True)

        return {'img': img_tensor, 'label': label}

    def __len__(self):
        return len(self.paths)

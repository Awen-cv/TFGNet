import os
import cv2
import math
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from models.basicsr.utils import img2tensor
from models.basicsr.data.transforms import augment
from models.basicsr.data import gaussian_kernels  # 添加导入
from models.basicsr.utils.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class SingleImageDataset(Dataset):
    def __init__(self, opt):
        super(SingleImageDataset, self).__init__()
        self.opt = opt
        self.gt_folder = opt['dataroot_gt']
        self.gt_size = opt.get('gt_size', 512)
        self.in_size = opt.get('in_size', 512)
        self.mean = opt.get('mean', [0.5, 0.5, 0.5])
        self.std = opt.get('std', [0.5, 0.5, 0.5])


        self.latent_gt_path = opt.get('latent_gt_path', None)
        if self.latent_gt_path:
            self.load_latent_gt = True
            self.latent_gt_dict = torch.load(self.latent_gt_path)
        else:
            self.load_latent_gt = False


        self.paths = sorted([os.path.join(self.gt_folder, f) for f in os.listdir(self.gt_folder) if f.endswith('.png')])


        self.blur_kernel_size = opt.get('blur_kernel_size', 41)
        self.kernel_list = opt.get('kernel_list', ['iso', 'aniso'])
        self.kernel_prob = opt.get('kernel_prob', [0.5, 0.5])
        self.blur_sigma = opt.get('blur_sigma', [1, 15])
        self.downsample_range = opt.get('downsample_range', [4, 30])
        self.noise_range = opt.get('noise_range', [0, 20])
        self.jpeg_range = opt.get('jpeg_range', [30, 80])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        gt_path = self.paths[index]
        img_gt = cv2.imread(gt_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
        name = os.path.basename(gt_path)[:-4]


        latent_gt = self.latent_gt_dict.get(name, None) if self.load_latent_gt else None
        if latent_gt is not None:
            latent_gt = torch.tensor(latent_gt).long()


        img_in = self.generate_low_quality(img_gt)


        img_in, img_gt = augment([img_in, img_gt], hflip=self.opt.get('use_hflip', True), rotation=False)


        img_in, img_gt = img2tensor([img_in, img_gt], bgr2rgb=True, float32=True)
        img_in = self.normalize(img_in)
        img_gt = self.normalize(img_gt)

        return {'in': img_in, 'gt': img_gt, 'gt_path': gt_path, 'latent_gt': latent_gt}

    def generate_low_quality(self, img):
        # Gaussian blur
        kernel = gaussian_kernels.random_mixed_kernels(
            self.kernel_list,
            self.kernel_prob,
            self.blur_kernel_size,
            self.blur_sigma,
            self.blur_sigma,
            [-math.pi, math.pi],
            noise_range=None
        )
        img = cv2.filter2D(img, -1, kernel)


        scale = np.random.uniform(self.downsample_range[0], self.downsample_range[1])
        h, w = img.shape[:2]
        img = cv2.resize(img, (int(w // scale), int(h // scale)), interpolation=cv2.INTER_LINEAR)


        noise_sigma = np.random.uniform(self.noise_range[0] / 255., self.noise_range[1] / 255.)
        noise = np.random.randn(*img.shape) * noise_sigma
        img = np.clip(img + noise, 0, 1)


        jpeg_quality = np.random.randint(self.jpeg_range[0], self.jpeg_range[1])
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
        _, encimg = cv2.imencode('.jpg', img * 255.0, encode_param)
        img = cv2.imdecode(encimg, 1) / 255.0


        img = cv2.resize(img, (self.in_size, self.in_size), interpolation=cv2.INTER_LINEAR)
        return img

    def normalize(self, img):
        img = (img - np.array(self.mean).reshape(3, 1, 1)) / np.array(self.std).reshape(3, 1, 1)
        return img

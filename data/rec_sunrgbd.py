import os
from collections import Counter

import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
from data import custom_transforms as tr
import numpy as np
import copy

class Rec_SUNRGBD:

    def __init__(self, cfg, split=None):
        self.cfg = cfg
        self.labeled = True
        self.ignore_label = -100
        self.split = split
        if split == 'train':
            self.data_dir = cfg.train_path
        elif split == 'test':
            self.data_dir = cfg.test_path
        self.class_weights = None
        self.class_weights_aux = None

        if self.labeled:
            self.classes, self.class_to_idx = self.find_classes(self.data_dir)
            self.int_to_class = dict(zip(range(len(self.classes)), self.classes))
            self.imgs = self.make_dataset()

            class_weights = np.array(list(Counter([i[1] for i in self.imgs]).values()))

            # enhance_weights = class_weights
            balance_weights = list(1 - class_weights / sum(class_weights))
            enhance_weights = list(class_weights)

            # ##### max
            # max_num = max(class_weights)
            # min_num = min(class_weights)
            # enhance_weights2 = list((class_weights - min_num + 0.01) / (max_num - min_num))

            if cfg.class_weights == 'enhance':
                self.class_weights = enhance_weights
            elif cfg.class_weights == 'balanced':
                self.class_weights = balance_weights
            else:
                self.class_weights = None

            # self.class_weights = list(max_num/ class_weights)
            #
            if cfg.class_weights_aux == 'enhance':
                self.class_weights_aux = enhance_weights
            elif cfg.class_weights_aux == 'balance':
                self.class_weights_aux = balance_weights
            else:
                self.class_weights_aux = None
        else:
            self.imgs = self.get_images()
            self.class_weights = None

    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def make_dataset(self):
        images = []
        dir = os.path.expanduser(self.data_dir)
        for target in sorted(os.listdir(dir)):
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self.class_to_idx[target])
                    images.append(item)
        return images

    def get_images(self):
        images = []
        dir = os.path.expanduser(self.data_dir)
        image_names = [d for d in os.listdir(dir)]
        for image_name in image_names:
            file = os.path.join(dir, image_name)
            images.append(file)
        return images

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        if self.labeled:
            img_path, label = self.imgs[index]
        else:
            img_path = self.imgs[index]

        img_name = os.path.basename(img_path)
        AB_conc = Image.open(img_path).convert('RGB')

        # split RGB and Depth as A and B
        w, h = AB_conc.size
        w2 = int(w / 2)
        if w2 > self.cfg.crop_size:
            A = AB_conc.crop((0, 0, w2, h)).resize((self.cfg.load_size, self.cfg.load_size), Image.BICUBIC)
            B = AB_conc.crop((w2, 0, w, h)).resize((self.cfg.load_size, self.cfg.load_size), Image.BICUBIC)
        else:
            A = AB_conc.crop((0, 0, w2, h))
            B = AB_conc.crop((w2, 0, w, h))

        if self.labeled:
            sample = {'image': A, 'label': label, 'path': img_path, 'name': img_name}
        else:
            sample = {'image': A}

        sample['depth'] = B

        if self.cfg.direction == 'BtoA':
            sample['image'], sample['depth'] = sample['depth'], sample['image']

        if self.split == 'train':
            return self.transform_tr(self.cfg, sample)
        else:
            return self.transform_test(self.cfg, sample)

    def transform_tr(self, cfg, sample):

        train_transforms = [
            tr.Resize(cfg.load_size),
            tr.RandomCrop(cfg.crop_size),
            tr.RandomHorizontalFlip(),
            tr.ToTensor(),
            tr.Normalize(mean=cfg.mean, std=cfg.std)
        ]
        if cfg.multi_scale:
            train_transforms.insert(-2, tr.MultiScale(size=(cfg.crop_size, cfg.crop_size), scale_times=cfg.multi_scale_num, keys=cfg.ms_keys))
        composed_transforms = transforms.Compose(train_transforms)
        return composed_transforms(sample)

    def transform_test(self, cfg, sample):

        composed_test = transforms.Compose([
            tr.Resize(cfg.load_size),
            tr.CenterCrop(cfg.crop_size),
            tr.ToTensor(),
            tr.Normalize(mean=cfg.mean, std=cfg.std)
        ])
        return composed_test(sample)

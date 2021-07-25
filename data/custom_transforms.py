import numbers
import random

import PIL.ImageEnhance as ImageEnhance
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
from torchvision.transforms import functional as F

class Resize(transforms.Resize):

    def __call__(self, sample):

        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.resize(sample[key], self.size, interpolation=Image.BICUBIC)

        return sample


class RandomCrop(transforms.RandomCrop):

    def __call__(self, sample):
        img = sample['image']

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (int((1 + self.size[1] - img.size[0]) / 2), 0))
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, int((1 + self.size[0] - img.size[1]) / 2)))

        i, j, h, w = self.get_params(img, self.size)

        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.crop(sample[key], i, j, h, w)

        return sample


class CenterCrop(transforms.CenterCrop):

    def __call__(self, sample):
        for key in sample.keys():
            if not isinstance(sample[key], Image.Image):
                continue
            sample[key] = F.center_crop(sample[key], self.size)

        return sample


class RandomHorizontalFlip(transforms.RandomHorizontalFlip):

    def __call__(self, sample):
        if random.random() > 0.5:
            for key in sample.keys():
                if not isinstance(sample[key], Image.Image):
                    continue
                sample[key] = F.hflip(sample[key])

        return sample


class MultiScale(object):

    def __init__(self, size, scale_times=5, keys=[]):
        assert keys
        self.keys = keys
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.scale_times = scale_times

    def __call__(self, sample):
        h = self.size[0]
        w = self.size[1]

        for key in self.keys:
            if key not in sample.keys():
                raise ValueError('multiscale keys not in sample keys!!!')
            item = sample[key]
            # if not isinstance(sample[key], Image.Image):
            #     continue
            sample['ms_' + key] = [F.resize(item, (h // pow(2, i), w // pow(2, i)), interpolation=Image.BILINEAR) for
                           i in range(self.scale_times)]
        return sample

class ToTensor(object):
    def __call__(self, sample):

        for key in sample.keys():
            if 'pil' in key:
                continue
            if isinstance(sample[key], list):
                sample[key] = [F.to_tensor(item) for item in sample[key]]
            elif not isinstance(sample[key], Image.Image):
                continue
            else:
                sample[key] = F.to_tensor(sample[key])
        return sample


class Normalize(transforms.Normalize):

    def __init__(self, mean, std):
        super().__init__(mean, std)

    def __call__(self, sample):

        for key in sample.keys():
            if 'pil' in key:
                continue
            if isinstance(sample[key], list):
                sample[key] = [F.normalize(item, self.mean, self.std) for item in sample[key]]
            elif not F._is_tensor_image(sample[key]):
                continue
            else:
                sample[key] = F.normalize(sample[key], self.mean, self.std)
        return sample

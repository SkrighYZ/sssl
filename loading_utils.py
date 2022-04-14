import numpy as np
from numpy.random import default_rng
import random
import os

from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from Stream51Dataset import Stream51Dataset


class RehearsalBatchSampler(torch.utils.data.Sampler):
    """
    A sampler that returns a generator obj30216ect which randomly samples from a list, that holds the indices that are
    eligible for rehearsal.
    The samples that are eligible for rehearsal grows over time, so we want it to be a 'generator' object and not an
    iterator object.
    """

    # See: https://github.com/pytorch/pytorch/issues/683
    def __init__(self, rehearsal_ixs, num_rehearsal_samples):
        # This makes sure that different workers have different randomness and don't end up returning the same data
        # item!
        self.rehearsal_ixs = rehearsal_ixs  # These are the samples which can be replayed. This list can grow over time.
        self.num_rehearsal_samples = num_rehearsal_samples

        self.rng = default_rng(seed=os.getpid())

    def __iter__(self):
        # We are returning a generator instead of an iterator, because the data points we want to sample from, differs
        # every time we loop through the data.
        # e.g., if we are seeing 100th sample, we may want to do a replay by sampling from 0-99 samples. But then,
        # when we see 101th sample, we want to replay from 0-100 samples instead of 0-99.

        while True:
            #ix = np.random.randint(0, len(self.rehearsal_ixs), self.num_rehearsal_samples)
            ix = self.rng.choice(len(self.rehearsal_ixs), self.num_rehearsal_samples, replace=False)
            yield np.array([self.rehearsal_ixs[_curr_ix] for _curr_ix in ix])

    def __len__(self):
        return 2 ** 64  # Returning a very large number because we do not want it to stop replaying.
        # The stop criteria must be defined in some other manner.


def get_stream_data_loaders(args, shuffle=False):

    if args.dataset == 'stream51':
        dataset = Stream51Dataset(args.images_dir, ordering=args.order, transform=Transform(), bbox_crop=True, ratio=1.10)
    else:
        raise NotImplementedError

    batch_size = args.batch_size if 'sliding' in args.model else 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)

    if 'sliding' in args.model:
        replay_sampler = None
        replay_loader = None
    else:
        replay_sampler = RehearsalBatchSampler(rehearsal_ixs=[], num_rehearsal_samples=args.batch_size-1)
        replay_loader = DataLoader(dataset, batch_sampler=replay_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return dataset, train_loader, replay_loader, replay_sampler



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(1, 1), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                             saturation=0.2, hue=0.1)],
            #     p=0.8
            # ),
            # transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(p=1.0),
            # Solarization(p=0.0),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(1, 1), interpolation=Image.BICUBIC),
            # transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomApply(
            #     [transforms.ColorJitter(brightness=0.4, contrast=0.4,
            #                             saturation=0.2, hue=0.1)],
            #     p=0.8
            # ),
            # transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(p=0.1),
            # Solarization(p=0.2),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

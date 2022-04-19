import numpy as np
from numpy.random import default_rng, randint
import random
import os
from tqdm import tqdm

from PIL import Image, ImageOps, ImageFilter
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from Stream51Dataset import Stream51Dataset

# This sampler is the way online sampler should be written in real life
# However, since PyTorch Dataloader uses multiple workers for speedup, they end up generating the same consecutive batches... not what we want
# Therefore, I can only simulate all the batches before each epoch in practice, implemented below this class (in RehearsalBatchSampler)
class LegacyRehearsalBatchSampler(torch.utils.data.Sampler):
    """
    A sampler that returns a generator object which randomly samples from memory, that holds the indices that are
    eligible for rehearsal.
    The samples that are eligible for rehearsal grows over time, so we want it to be a 'generator' object and not an
    iterator object.
    """

    def __init__(self, stm_span, num_rehearsal_samples, long_term_mem=[], short_term_mem=[]):
        self.long_term_mem = long_term_mem  
        self.short_term_mem = short_term_mem
        self.stm_time_passed = None  # need to call init_memory()
        self.stm_span = stm_span

        self.num_rehearsal_samples = num_rehearsal_samples

        self.rng = default_rng(seed=os.getpid())

    def __iter__(self):
        while True:
            rehearsal_idxs = self.long_term_mem + self.short_term_mem
            if self.num_rehearsal_samples == len(rehearsal_idxs):
                #print(rehearsal_idxs)
                yield np.array(rehearsal_idxs)
            else:
                ix = self.rng.choice(len(rehearsal_idxs), self.num_rehearsal_samples, replace=False)
                yield np.array([rehearsal_idxs[_curr_ix] for _curr_ix in ix])

    def __len__(self):
        # Returning a very large number because we do not want it to stop replaying.
        # The stop criteria must be defined in some other manner.
        return 2 ** 64  

    def init_memory(self, ltm_size, stm_size):
        self.long_term_mem = list(range(ltm_size))
        self.short_term_mem = list(range(stm_size))
        self.stm_time_passed = np.arange(len(self.short_term_mem)-1, -1, -1)

    def update_memory(self, t, update_ltm=True):
        # Assume long term memory is larger
        if update_ltm:
            # Update long term memory
            replace_idx = randint(0, t+1)
            if replace_idx < len(self.long_term_mem):
                self.long_term_mem[replace_idx] = t
            replace_idx = randint(0, t+1)

        # Update short term memory
        replace_idx = randint(0, self.stm_span+1)
        if replace_idx < len(self.short_term_mem):
            if np.max(self.stm_time_passed) > self.stm_span:
                replace_idx = np.argmax(self.stm_time_passed)
            self.short_term_mem[replace_idx] = t
            self.stm_time_passed[replace_idx] = 0
        self.stm_time_passed += 1



class RehearsalBatchSampler(torch.utils.data.Sampler):
    """
    A sampler that returns a generator object which randomly samples from memory, that holds the indices that are
    eligible for rehearsal.
    """

    def __init__(self, stm_span, num_rehearsal_samples, long_term_mem=[], short_term_mem=[]):
        self.long_term_mem = long_term_mem  
        self.short_term_mem = short_term_mem
        self.stm_time_passed = None  # need to call init_memory()
        self.stm_span = stm_span

        self.num_rehearsal_samples = num_rehearsal_samples
        self.batches = None  # Need to call simulate_batches()

        self.rng = default_rng(seed=os.getpid())

    def __iter__(self):
        for batch_idx in range(self.batches.shape[0]):
            yield self.batches[batch_idx]

    def __len__(self):
        return self.batches.shape[0]

    def init_memory(self, ltm_size, stm_size):
        self.long_term_mem = list(range(ltm_size))
        self.short_term_mem = list(range(stm_size))
        self.stm_time_passed = np.arange(len(self.short_term_mem)-1, -1, -1)

    def simulate_batches(self, ltm_size, stm_size, batch_size, num_examples):

        self.batches = np.zeros((num_examples//batch_size+1, batch_size), dtype=int)
        curr = 0
        for t in tqdm(range(num_examples)):
            if t < stm_size:
                continue
            elif t < ltm_size:
                self.update_memory(t, update_ltm=False)
            else:
                self.update_memory(t, update_ltm=True)

            if (t+1) % batch_size == 0:
                rehearsal_idxs = self.long_term_mem + self.short_term_mem
                if self.num_rehearsal_samples == len(rehearsal_idxs):
                    batch = np.array(rehearsal_idxs)
                else:
                	# Use all samples in stm and randomly select samples in ltm
                    ix = self.rng.choice(len(self.long_term_mem), self.num_rehearsal_samples, replace=False)
            		batch = np.array([self.long_term_mem[_curr_ix] for _curr_ix in ix] + self.short_term_mem)       
                self.batches[curr, :] = batch
                curr += 1
        
        # Last batch
        rehearsal_idxs = self.long_term_mem + self.short_term_mem
        if self.num_rehearsal_samples == len(rehearsal_idxs):
            batch = np.array(rehearsal_idxs)
        else:
        	# Use all samples in stm and randomly select samples in ltm
            ix = self.rng.choice(len(self.long_term_mem), self.num_rehearsal_samples, replace=False)
            batch = np.array([self.long_term_mem[_curr_ix] for _curr_ix in ix] + self.short_term_mem)
        self.batches[curr, :] = batch

        print(self.batches[:2, :])
        print(self.batches[-2:, :])

    def update_memory(self, t, update_ltm=True):
        # Assume long term memory is larger
        if update_ltm:
            # Update long term memory
            replace_idx = randint(0, t+1)
            if replace_idx < len(self.long_term_mem):
                self.long_term_mem[replace_idx] = t
            replace_idx = randint(0, t+1)

        # Update short term memory
        replace_idx = randint(0, self.stm_span+1)
        if replace_idx < len(self.short_term_mem):
            if np.max(self.stm_time_passed) > self.stm_span:
                replace_idx = np.argmax(self.stm_time_passed)
            self.short_term_mem[replace_idx] = t
            self.stm_time_passed[replace_idx] = 0
        self.stm_time_passed += 1



def get_stream_data_loaders(args):

    if args.dataset == 'stream51':
        if args.model == 'sliding_supervised':
            dataset = Stream51Dataset(args.images_dir, ordering=args.order, transform=Transform(), bbox_crop=True, ratio=1.10)
        else:
            dataset = Stream51Dataset(args.images_dir, ordering=args.order, transform=SSLTransform(), bbox_crop=True, ratio=1.10)
    else:
        raise NotImplementedError

    shuffle = True if args.order == 'iid' else False

    if 'sliding' in args.model:
        replay_sampler  = None
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.num_workers, pin_memory=True)
    else:
        replay_sampler = RehearsalBatchSampler(stm_span=args.stm_span, num_rehearsal_samples=args.batch_size)
        train_loader = DataLoader(dataset, batch_sampler=replay_sampler, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return dataset, train_loader, replay_sampler


def get_finetune_data_loaders(args):

    if args.dataset == 'stream51':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64, scale=(0.5, 1), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
                transforms.Resize(64, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        trainset = Stream51Dataset(args.images_dir, test=False, ordering=args.order, transform=train_transform, bbox_crop=True, ratio=1.10)
        testset = Stream51Dataset(args.images_dir, test=True, ordering=args.order, transform=test_transform, bbox_crop=True, ratio=1.10)
    
    elif args.dataset == 'tinyimagenet':
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64, interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        test_transform = transforms.Compose([
                transforms.Resize(64, interpolation=Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ])
        trainset = ImageFolder(root=args.images_dir / 'train', transform=train_transform)
        testset = ImageFolder(root=args.images_dir / 'val_restruc', transform=test_transform)
    
    else:
        raise NotImplementedError

    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader



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
                transforms.RandomResizedCrop(64, scale=(0.5, 1), interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

    def __call__(self, x):
        y = self.transform(x)
        return y

class SSLTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(64, scale=(0.5, 1), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

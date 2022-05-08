import os
import random
import json
from PIL import Image
import torch.utils.data as data
import copy


def instance_ordering(data_list):
    # organize data by video
    total_videos = 0
    new_data_list = []
    temp_video = []
    v2ex_mapping = []
    temp_video_idx = []
    for x_i, x in enumerate(data_list):
        if x[3] == 0:
            new_data_list.append(temp_video)
            temp_video = [x]
            v2ex_mapping.append(temp_video_idx)
            temp_video_idx = [x_i]
            total_videos += 1
        else:
            temp_video.append(x)
            temp_video_idx.append(x_i)
    new_data_list.append(temp_video)
    new_data_list = new_data_list[1:]
    v2ex_mapping.append(temp_video_idx)
    v2ex_mapping = v2ex_mapping[1:]
    
    # shuffle videos
    v2v_mapping = list(range(len(new_data_list)))
    random.shuffle(v2v_mapping)
    new_data_list = [new_data_list[i] for i in v2v_mapping]
    
    # reorganize by clip
    data_list = []
    ex2ex_mapping = []
    for v_i, v in enumerate(new_data_list):
        for x_i, x in enumerate(v):
            data_list.append(x)
            ex2ex_mapping.append(v2ex_mapping[v2v_mapping[v_i]][x_i])

    return data_list, ex2ex_mapping


class Stream51Dataset(data.Dataset):
    """Stream-51 Dataset Object
    Args:
        root (string): Root directory path of dataset.
        train (bool): load either training set (True) or test set (False) (default: True)
        ordering (string): desired ordering for training dataset: 'instance',
            'class_instance', 'iid', or 'class_iid' (ignored for test dataset)
            (default: None)
        transform: A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform: A function/transform that takes
            in the target and transforms it.
        bbox_crop: crop images to object bounding box (default: True)
        ratio: padding for bbox crop (default: 1.10)
     Attributes:
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, test=False, ordering=None, transform=None, bbox_crop=True, ratio=1.10):

        if test:
            self.samples = json.load(open(os.path.join(root, 'Stream-51_meta_test.json')))
        else:
            self.samples = json.load(open(os.path.join(root, 'Stream-51_meta_train.json')))
        self.targets = [s[0] for s in self.samples]
        
        # First frames of each video clip
        self.shot_bounds = []
        for i in range(len(self.samples)):
            if self.samples[i][3] == 0:
                self.shot_bounds += [1]
            else:
                self.shot_bounds += [0]

        self.ordering = ordering

        self.root = root
        self.loader = default_loader

        self.transform = transform

        self.bbox_crop = bbox_crop
        self.ratio = ratio

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        index = int(index)
        fpath, target = self.samples[index][-1], self.targets[index]
        sample = self.loader(os.path.join(self.root, fpath))
        if self.bbox_crop:
            bbox = self.samples[index][-2]
            cw = bbox[0] - bbox[1];
            ch = bbox[2] - bbox[3];
            center = [int(bbox[1] + cw / 2), int(bbox[3] + ch / 2)]
            bbox = [min([int(center[0] + (cw * self.ratio / 2)), sample.size[0]]),
                    max([int(center[0] - (cw * self.ratio / 2)), 0]),
                    min([int(center[1] + (ch * self.ratio / 2)), sample.size[1]]),
                    max([int(center[1] - (ch * self.ratio / 2)), 0])]
            sample = sample.crop((bbox[1],
                                  bbox[3],
                                  bbox[0],
                                  bbox[2]))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def shuffle(self):
        """
        samples
        for train: [class_id, clip_num, video_num, frame_num, bbox, file_loc]
        for test: [class_id, bbox, file_loc]
        """
        ex2ex_mapping = None
        if self.ordering == 'iid':
            # shuffle all data
            random.shuffle(self.samples)
            self.targets = [s[0] for s in self.samples]
            self.shot_bounds = []
            for i in range(len(self.samples)):
                if self.samples[i][3] == 0:
                    self.shot_bounds += [1]
                else:
                    self.shot_bounds += [0]
        elif self.ordering == 'instance':
            self.samples, ex2ex_mapping = instance_ordering(self.samples)
            self.targets = [s[0] for s in self.samples]
            self.shot_bounds = []
            for i in range(len(self.samples)):
                if self.samples[i][3] == 0:
                    self.shot_bounds += [1]
                else:
                    self.shot_bounds += [0]
        else:
            raise ValueError('dataset ordering must be one of: "iid" or "instance"')

        return ex2ex_mapping


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def default_loader(path):
    return pil_loader(path)

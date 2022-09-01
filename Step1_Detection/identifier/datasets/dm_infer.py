from __future__ import absolute_import
from __future__ import print_function
import pickle
from PIL.Image import merge
from detectron2.structures import boxes

from torch.utils.data import DataLoader

from . import init_imgreid_dataset
from utils.torch_func import get_mean_and_std, calculate_mean_and_std
from PIL import Image
import os.path as osp
from torch.utils.data import Dataset
from torchvision.transforms import *
from collections import defaultdict
import numpy as np
import copy
import random

from torch.utils.data.sampler import Sampler, RandomSampler


def build_transforms(height,
                     width,
                     **kwargs):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    normalize = Normalize(mean=imagenet_mean, std=imagenet_std)

    transform_test = Compose([
        Resize((height, width)),
        ToTensor(),
        normalize,
    ])

    return transform_test


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError('{} does not exist'.format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print('IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'.format(
                img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class BaseDataManager(object):

    def __init__(self,
                 use_gpu,
                 test_set,
                 root='imgs',
                 height=128,
                 width=256,
                 test_batch_size=100,
                 workers=4,
                 # number of instances per identity (for RandomIdentitySampler)
                 num_instances=4,
                 **kwargs
                 ):
        self.use_gpu = use_gpu
        self.test_set = test_set
        self.root = root
        self.height = height
        self.width = width
        self.test_batch_size = test_batch_size
        self.workers = workers
        self.num_instances = num_instances

        transform_test = build_transforms(self.height, self.width)
        self.transform_test = transform_test

    def return_dataloaders(self):
        """
        Return trainloader and testloader dictionary
        """
        return self.testloader_dict


class ImageDataManager(BaseDataManager):
    def __init__(self,
                 use_gpu,
                 test_set,
                 **kwargs
                 ):
        super(ImageDataManager, self).__init__(use_gpu, test_set, **kwargs)

        print('=> Initializing TEST datasets')
        self.testdataset_dict = {"query": None, "test": None}
        self.testloader_dict = {"query": None, "test": None}

        dataset = init_imgreid_dataset(
            root=self.root, name=test_set)

        self.testloader_dict['query'] = DataLoader(
            ImageDataset(dataset.query, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

        self.testloader_dict['test'] = DataLoader(
            ImageDataset(dataset.test, transform=self.transform_test),
            batch_size=self.test_batch_size, shuffle=False, num_workers=self.workers,
            pin_memory=self.use_gpu, drop_last=False
        )

        self.testdataset_dict['query'] = dataset.query
        self.testdataset_dict['test'] = dataset.test


class PickleDataManager(BaseDataManager):
    def __init__(self, feature_type="reid", dataset="CAMPUS", scenes="Auditorium", merge_mc=True, cvt_to_MOT_format=True, **kwargs) -> None:
        super(PickleDataManager, self).__init__(use_gpu=True, **kwargs)

        # print('=> Initializing Pickle TEST datasets')
        self.testloader_dict = {"query": None, "test": None}
        self.length_dict = {("MCT", "Dataset1"): 24000, ("MCT", "Dataset2"): 24000, ("MCT", "Dataset3"): 5251, ("MCT", "Dataset4"): 36001,
        (".", "PETS09"): 795, ("EPFL", "Laboratory"): 2955, ("EPFL", "Basketball"): 9368, ("EPFL", "Passageway"): 2500, ("EPFL", "Terrace"): 5010,
        ("EPFL", "Campus"): 5884, ("CAMPUS", "Parkinglot"): 6478, ("CAMPUS", "Auditorium"): 5458, ("CAMPUS", "Garden1"): 2983, ("CAMPUS", "Garden2"): 6000,
        ("aic", "S02"): 2110, ("aic", "S05"): 4299}

        self.dataset = dataset
        self.scenes = scenes
        self.feature_type = feature_type
        self.datapath = osp.join(self.root, self.dataset, self.scenes)
        self.cvt_to_MOT_format = cvt_to_MOT_format
        self.merge_mc = merge_mc

        self.check_before_run()

        # with open(osp.join(self.datapath, feature_type + "_feats.pkl"), "rb") as f:
        #     self.pickle = pickle.load(f)

        with open(osp.join(self.datapath, "bboxes.pkl"), "rb") as fi:
            pkl_bboxes = pickle.load(fi)

        car_ids = {}
        cur_id = 0
        cur_frid = 0
        _query = []
        _test = []

        if self.merge_mc:
            merged_gt_fo = open(
                osp.join(self.datapath, "gt_" + scenes + "-merged.txt"), "w")

        for idx, cam in enumerate(sorted(pkl_bboxes.keys())):
            if self.cvt_to_MOT_format:
                fo = open(osp.join(self.datapath,
                                   "gt_" + cam[:-3] + "txt"), "w")

            frames = sorted(list(pkl_bboxes[cam].keys()))
            for frame in frames:
                for _id in sorted(pkl_bboxes[cam][frame].keys()):
                    bbox = ",".join(
                        list(map(str, pkl_bboxes[cam][frame][_id])))
                    # if not("_".join((cam, _id)) in car_ids.keys()):
                    #     # img, pid, camid, img_path
                    #     car_ids["_".join((cam, _id))] = cur_id
                    #     _query.append(
                    #         (self.pickle[cam][frame][_id], car_ids["_".join((cam, _id))], cam, frame, bbox))
                    #     cur_id += 1
                    # else:
                    #     _test.append((self.pickle[cam][frame][_id], car_ids["_".join(
                    #         (cam, _id))], cam, frame, bbox))
                    if self.cvt_to_MOT_format:
                        fo.write(
                            ",".join((str(frame), str(int(_id)), bbox, "1,-1,-1,-1\n")))
                        if self.merge_mc:
                            merged_gt_fo.write(
                                ",".join((str(frame + cur_frid), str(int(_id)), bbox, "1,-1,-1,-1\n")))
            cur_frid = self.length_dict[(dataset, scenes)]*(idx + 1)

            if self.cvt_to_MOT_format:
                fo.close()

        if self.merge_mc:
            merged_gt_fo.close()

        # print(cur_id)
        self.testloader_dict['query'] = _query
        self.testloader_dict['test'] = _test

    def check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.datapath):
            raise RuntimeError('"{}" is not available'.format(self.datapath))

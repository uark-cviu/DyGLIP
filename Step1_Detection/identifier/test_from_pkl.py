from __future__ import print_function
from __future__ import division

import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import warnings
from identifier.train import test
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from opts import argument_parser, testset_kwargs
from datasets.dm_infer import ImageDataManager, PickleDataManager
import models
from losses import CrossEntropyLoss, TripletLoss, DeepSupervision
from utils.io import check_isfile
from utils.avgmeter import AverageMeter
from utils.log import Logger, RankLogger
from utils.torch_func import count_num_param, load_pretrained_weights
from utils.seed import set_random_seed
from postprocess.postprocess import calc_reid, update_output

# global variables
parser = argument_parser()
args = parser.parse_args()


def main():
    global args
    if not args.use_avai_gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
    use_gpu = torch.cuda.is_available()
    if args.use_cpu:
        use_gpu = False

    if use_gpu:
        print('Currently using GPU {}'.format(args.gpu_devices))
        cudnn.benchmark = True
    else:
        warnings.warn(
            'Currently using CPU, however, GPU is highly recommended')

    # print('Initializing image data manager')

    # for a, b in [("CAMPUS", "Auditorium"), ("CAMPUS", "Garden1"), ("CAMPUS", "Garden2"), ("CAMPUS", "Parkinglot"),
    #              ("EPFL", "Basketball"), ("EPFL", "Campus"), ("EPFL", "Laboratory"), ("EPFL", "Passageway"), ("EPFL", "Terrace"), (".", "PETS09"),
    #              ("MCT", "Dataset1"), ("MCT", "Dataset2"), ("MCT", "Dataset3"), ("MCT", "Dataset4")]:
    for a, b in [("aic", "S02"), ("aic", "S05")]:
        print(a, b)
        dm = PickleDataManager(dataset=a, scenes=b, **testset_kwargs(args))
        testloader_dict = dm.return_dataloaders()

        # print('Matching {} ...'.format(args.test_set))
        # queryloader = testloader_dict['query']
        # galleryloader = testloader_dict['test']
        # # test(model, queryloader, galleryloader, use_gpu)
        # distmat, q_pids, g_pids, q_camids, g_camids = run(
        #     queryloader, galleryloader, use_gpu, return_distmat=True)
        # np.savez(args.save_npy, distmat=distmat, q_pids=q_pids,
        #          g_pids=g_pids, q_camids=q_camids, g_camids=g_camids)
        # result = np.load(args.save_npy)
        # reid_dict, rm_dict = calc_reid(result)
        # # print(rm_dict)

        # update_output(dm, reid_dict, rm_dict)
        # print()


def run(queryloader, galleryloader, use_gpu, ranks=[1, 5, 10, 20], return_distmat=False):

    qf, q_pids, q_camids = [], [], []
    for batch_idx, (features, pids, camids, _, _) in enumerate(queryloader):

        qf.append(features)
        q_pids.append(pids)
        q_camids.append(camids)

    qf = torch.cat(qf, 0)
    q_pids = np.asarray(q_pids)
    q_camids = np.asarray(q_camids)

    # print('Extracted features for query set, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

    gf, g_pids, g_camids = [], [], []
    for batch_idx, (features, pids, camids, _, _) in enumerate(galleryloader):

        gf.append(features)
        g_pids.append(pids)
        g_camids.append(camids)
    gf = torch.cat(gf, 0)
    g_pids = np.asarray(g_pids)
    g_camids = np.asarray(g_camids)

    # print('Extracted features for gallery set, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))

    m, n = qf.size(0), gf.size(0)
    distmat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
        torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    distmat.addmm_(1, -2, qf, gf.t())
    distmat = distmat.numpy()

    return distmat, q_pids, g_pids, q_camids, g_camids


if __name__ == '__main__':
    main()

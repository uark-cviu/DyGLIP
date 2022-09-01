"""
Implementation of non-negative matrix factorization 
"""

import pickle
from operator import sub
import numpy as np

from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path
import pandas as pd
import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes


def genTestingS(N, k):
    A = np.zeros((N, k))

    for i in range(N):
        j = np.random.randint(k)
        A[i, j] = 1

    S = np.matmul(A, A.transpose())
    return S, A


def binarize(A_real):
    A = np.zeros(A_real.shape)
    J = np.argmax(A_real, 1)
    for i in range(A.shape[0]):
        A[i, J[i]] = 1
    return A


def objective(S, A):
    return np.linalg.norm(S - np.matmul(A, A.transpose()))


def NMF(S, k):
    """
    Solve the problem min_{A} ||S - A'A|| s.t. A1_K = 1_N

    """
    max_iter = 1000
    N = S.shape[0]
    A = np.random.rand(N, int(k))
    # print('----Initialized A matrix-----')
    # print(A)
    v1 = np.ones((k, 1))
    v1T = v1.transpose()

    v2 = np.ones((N, 1))
    v2T = v2.transpose()

    alpha = 10  # 10000
    alpha_moment = 0.1
    # print(alpha_moment)
    for iteration in range(max_iter):
        cost = objective(S, A)

        C = 4 * np.matmul(S, A) + 2*alpha*np.matmul(v2, v1.transpose())
        D = 4 * np.matmul(np.matmul(A, A.transpose()), A) + \
            2 * alpha * np.matmul(np.matmul(A, v1), v1T)
        E = np.divide(C, D)
        E = np.sqrt(E)
        A = np.multiply(A, E)

        if iteration % 50 == 0:
            alpha *= alpha_moment
            #print('iteration = {}, alpha = {}, cost = {}'.format(iteration, alpha, cost))
    return A


"""
N = 15
k = 6
import random
random.seed(10)
S, A_truth = genTestingS(N, k)

from datetime import datetime
random.seed(datetime.now())

print('----Input S matrix-----')
print(S)
A = NMF(S, k) 
A_bin = binarize(A)
print('----Real values of solutions-----')
print(A)
print('---- Binarize solutions ------')
print(A_bin)
print('----Ground truth binary matrix----')
print (A_truth)
print('-----Diff------')
print(A_truth - A_bin)
print('Total Wrong = {} / {} elements'.format(np.sum(np.abs(A_truth - A_bin))  / 2, N))
"""


# Load real data
# dataset = 'CAMPUS'
# subset = 'Auditorium' # 'Garden1' or 'Garden2' or 'Parkinglot' or 'Auditorium'

# dataset = 'PETS09'
# subset = '' # no subset

# dataset = 'EPFL'
# subset = 'Passageway' # no subset 'Basketball', 'Campus', 'Laboratory', 'Passageway', 'Terrace'

datasets = ["MCT"]
subsets = {'CAMPUS': ['Garden1', 'Garden2', 'Parkinglot', 'Auditorium'], '.': [
    'PETS09'], 'EPFL':  ['Basketball', 'Campus', 'Laboratory', 'Passageway', 'Terrace'],
    'MCT': ['Dataset1', "Dataset2", "Dataset3", "Dataset4"]}

# features = ['reid', 'maskrcnn']   # maskrcnn or reid


# datasets = ['MCT']
# subsets = {'MCT':  ['Dataset1', "Dataset2", "Dataset3", "Dataset4"]}
features = ['reid']

length_dict = {("MCT", "Dataset1"): 24000, ("MCT", "Dataset2"): 24000, ("MCT", "Dataset3"): 5251, ("MCT", "Dataset4"): 36001,
(".", "PETS09"): 795, ("EPFL", "Laboratory"): 2955, ("EPFL", "Basketball"): 9368, ("EPFL", "Passageway"): 2500, ("EPFL", "Terrace"): 5010,
("EPFL", "Campus"): 5884, ("CAMPUS", "Parkinglot"): 6478, ("CAMPUS", "Auditorium"): 5458, ("CAMPUS", "Garden1"): 2983, ("CAMPUS", "Garden2"): 6000}


def run_nmf(dataset, subset, feature):
    _datapath = './exp_bigger'
    feature_path = os.path.join(
        _datapath, dataset, subset, feature + '_feats.pkl')
    bboxes_path = os.path.join(_datapath, dataset, subset, 'bboxes.pkl')

    output_dir = _datapath

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    data = pickle.load(open(feature_path, "rb"))
    data_bbox = pickle.load(open(bboxes_path, "rb"))
    # print(data)

    # compute matrix S
    all_features = None
    max_track = 0
    all_pid = []
    for v in sorted(data.keys()):
        # print('Loading from {}'.format(v))
        # print(len(data[v]))
        pid = dict()
        features = []

        for fid in sorted(data[v].keys()):
            # print('Loading from frame-{} in video {}'.format(fid, v))
            # print(len(data[v][fid]))

            for id in sorted(data[v][fid].keys()):
                if id not in pid:
                    pid[id] = len(pid)
                    features.append([])
                feat = np.asarray(data[v][fid][id])
                feat = feat / np.linalg.norm(feat)

                #feat = 1 / (float(id) + 1)
                features[pid[id]].append(feat)

        # compute mean features
        for i in range(len(features)):
            feat = np.squeeze(np.asarray(features[i]))
            features[i] = np.mean(feat, axis=0)

        all_pid.append(pid)

        features = np.asarray(features)
        # print(features.shape)

        if features.shape[0] > max_track:
            max_track = features.shape[0]

        if all_features is None:
            all_features = features
        else:
            all_features = np.concatenate((all_features, features), axis=0)

    #all_features = np.expand_dims(np.asarray(all_features), axis=1)
    all_features = np.asarray(all_features)
    # print(all_features.shape)

    total_tracklets = 0

    for v in range(len(all_pid)):
        total_tracklets += len(all_pid[v])

    S = np.zeros((total_tracklets, total_tracklets))
    u = 0
    v = 0
    id_list = []
    for vi in range(len(all_pid)):
        for ti in all_pid[vi].keys():
            v = 0
            id_list.append(ti)

            for vj in range(len(all_pid)):
                for tj in all_pid[vj].keys():

                    if vi == vj:
                        S[u, v] = 0
                    else:
                        if ti == tj:
                            S[u, v] = 1
                        else:
                            S[u, v] = 0
                    v += 1

            u += 1

    # S = np.dot(all_features, np.transpose(all_features))

    # Estimate no. of targets
    u, s, vh = np.linalg.svd(S, full_matrices=True)
    # print(s)
    beta = 0.9  # 1.05 - GT #0.9 - predict
    K1 = np.cumsum(s >= beta)[-1]
    K = K1  # max_track # use ground truth
    # print('Estimated no. of targets: {}'.format(K1))
    # print('Ground truth no. of targets: {}'.format(max_track))

    # Compute matrix A
    A = NMF(S, K)
    A_bin = binarize(A)

    count_correct = 0

    for k in range(K):
        matched_tracks = np.where(A_bin[:, k] == 1)[0]

        if matched_tracks.shape[0] > 0:
            matched_id = id_list[matched_tracks[0]]
            count = 0
            for t in matched_tracks:
                if matched_id == id_list[t]:
                    count += 1

            if count == len(matched_tracks) and count != 0:
                count_correct += 1

    vi = 0
    gtfiles = []
    tsfiles = []
    frames = []
    cur_frid = 0
    datapath = os.path.join(_datapath, dataset, subset)
    merged_pred_fo = open(os.path.join(
        datapath, "pred_" + datapath.split("/")[-1] + "-merged.txt"), "w")

    for _idx, v in enumerate(sorted(data.keys())):

        # if not os.path.exists(os.path.join(output_dir, dataset, subset)):
        #     os.makedirs(os.path.join(output_dir, dataset, subset))

        gtfiles.append(os.path.join(datapath, "gt_" + v[:-3] + "txt"))
        tsfiles.append(os.path.join(datapath, "pred_" + v[:-3] + "txt"))
        f = open(os.path.join(datapath, "pred_" + v[:-3] + 'txt'), 'wt')

        pid = all_pid[vi]

        for fid in sorted(data[v].keys()):

            for id in sorted(data[v][fid].keys()):
                if id not in pid.keys():
                    continue
                idx = pid[id]
                assigned_id = np.where(A_bin[idx, :] == 1)[0][0]
                bbox = data_bbox[v][fid][id]
                f.write('{},{},{},{},{},{},1,-1,-1,-1\n'.format(fid,
                                                                str(assigned_id), bbox[0], bbox[1], bbox[2], bbox[3]))
                merged_pred_fo.write(",".join(
                    (str(fid + cur_frid), str(assigned_id), str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3]), "1,-1,-1,-1\n")))

        cur_frid = length_dict[(dataset, subset)]*(_idx + 1)

        f.close()
        vi += 1
    merged_pred_fo.close()

    gt = OrderedDict([(Path(f).parts[-1].split("_")[-1], mm.io.loadtxt(f,
                                                                       fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([(Path(f).parts[-1].split("_")[-1],
                       mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles])

    mh = mm.metrics.create()
    accs, names = compare_dataframes(gt, ts)

    metrics = list(mm.metrics.motchallenge_metrics)
    summary = mh.compute_many(
        accs, names=names, metrics=metrics, generate_overall=True)

    gtfiles = [(os.path.join(datapath, "gt_" +
                             datapath.split("/")[-1] + "-merged.txt"))]
    tsfiles = [(os.path.join(datapath, "pred_" +
                             datapath.split("/")[-1] + "-merged.txt"))]

    gt = OrderedDict([("MCMT", mm.io.loadtxt(f,
                                             fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
    ts = OrderedDict([("MCMT",
                       mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles])

    accs, names = compare_dataframes(gt, ts)

    summary = pd.concat([summary] + [mh.compute_many(
        accs, names=names, metrics=metrics)])
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))

    # print('Correctly assigned {}/{} targets'.format(count_correct, K))
    # print(A_bin)


for dataset in datasets:
    for subset in sorted(subsets[dataset]):
        print(dataset, subset)
        for feature in features:
            run_nmf(dataset, subset, feature)
        print()

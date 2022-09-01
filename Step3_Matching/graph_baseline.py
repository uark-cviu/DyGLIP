from __future__ import division
from __future__ import print_function

import json
import os
from os import curdir
import os.path as osp
import time
import numpy as np
import pickle
import argparse
import torch
from scipy.spatial import distance

from collections import OrderedDict
import glob
from pathlib import Path

from numpy.lib.function_base import append
import pandas as pd
import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--root', type=str, default='./exp/imgs',
                    help='root path to data directory')

args = parser.parse_args()

length_dict = {("MCT", "Dataset1"): 24000, ("MCT", "Dataset2"): 24000, ("MCT", "Dataset3"): 5251, ("MCT", "Dataset4"): 36001,
(".", "PETS09"): 795, ("EPFL", "Laboratory"): 2955, ("EPFL", "Basketball"): 9368, ("EPFL", "Passageway"): 2500, ("EPFL", "Terrace"): 5010,
("EPFL", "Campus"): 5884, ("CAMPUS", "Parkinglot"): 6478, ("CAMPUS", "Auditorium"): 5458, ("CAMPUS", "Garden1"): 2983, ("CAMPUS", "Garden2"): 6000}


def main(args, overlapping=False, alpha=0.5, inter_dist_thresh=15, intra_dist_thresh=2):
    data_folders = [("CAMPUS", "Auditorium")]

    # data_folders = [("CAMPUS", "Auditorium"), ("CAMPUS", "Garden1"), ("CAMPUS", "Garden2"), ("CAMPUS", "Parkinglot"),
    #             ("EPFL", "Basketball"), ("EPFL", "Campus"), ("EPFL",
    #                                                             "Laboratory"), ("EPFL", "Passageway"), ("EPFL", "Terrace"),
    #             (".", "PETS09"), ("MCT", "Dataset1"), ("MCT", "Dataset2"), ("MCT", "Dataset3"), ("MCT", "Dataset4")]

    # data_folders = [(".", "PETS09")]
    for a, b in data_folders:
        graphs = pickle.load(
            open(osp.join(args.root, a, b, "graphs.pkl"), "rb"))
        bboxes = pickle.load(
            open(osp.join(args.root, a, b, "bboxes.pkl"), "rb"))
        feats = pickle.load(
            open(osp.join(args.root, a, b, "reid_feats.pkl"), "rb"))
        for alpha in np.arange(0.5, 0.6, 0.2):
            for inter_dist_thresh in np.arange(6.0, 7.0, 2):
                for intra_dist_thresh in np.arange(4.0, 5.0, 2):
                    print(a, b, "alpha", alpha, "inter_dist_thresh",
                          inter_dist_thresh, "intra_dist_thresh", intra_dist_thresh)
                    gtfiles = []
                    tsfiles = []
                    fos = [open(osp.join(args.root, a, b, "pred_" + video[:-3] + "txt"), "w")
                           for video in sorted(feats.keys())]
                    times = []
                    time_steps = []
                    for video in sorted(feats.keys()):
                        times.append(0 if not len(
                            times) else len(times)*length_dict[(a, b)])
                        for time_step in feats[video].keys():
                            _ = time_steps.append(
                                time_step) if time_step not in time_steps else None
                    time_steps = sorted(time_steps)
                    merged = open(osp.join(args.root, a, b,
                                           "pred_" + b + "-merged.txt"), "w")
                    global_feats = {}
                    global_ids = {}
                    time_step_id = -1
                    for graph in graphs:
                        time_step_id += 1
                        time_step = time_steps[time_step_id]
                        new_nodes = []
                        for node in sorted(list(graph.nodes())):
                            video = "_".join(node.split("_")[0:-2])
                            track = node.split("_")[-2]
                            if node not in global_feats.keys():
                                new_nodes.append(node)
                                global_feats[node] = feats[video][time_step][track].detach(
                                ).numpy().ravel()
                                # global_feats[node] /= np.linalg.norm(global_feats[node])
                            elif (time_step in feats[video].keys() and track in feats[video][time_step].keys()):
                                feat = feats[video][time_step][track].detach(
                                ).numpy().ravel()
                                # feat /= np.linalg.norm(feat)
                                global_feats[node] = alpha*feat + \
                                    (1.0 - alpha)*global_feats[node]
                        if len(new_nodes):
                            if not (time_step_id == 0 and overlapping == False):
                                xA = np.array([global_feats[node] for node in sorted(
                                    global_feats.keys()) if node not in new_nodes])
                                xB = np.array([global_feats[node]
                                               for node in new_nodes])
                                dist = distance.cdist(xA, xB)
                                pairs = np.argmin(dist, axis=0)
                                assert len(pairs) == len(xB)
                                pairs_dist = np.amin(dist, axis=0)
                                assert len(pairs_dist) == len(xB)

                                assigned_nodes = [node for node in sorted(
                                    global_feats.keys()) if node not in new_nodes]
                                for i, (p, d) in enumerate(zip(pairs, pairs_dist)):
                                    new_node = new_nodes[i]
                                    assigned_node = assigned_nodes[pairs[i]]

                                    if d <= inter_dist_thresh:
                                        global_ids[new_node] = global_ids[assigned_node]
                                    else:
                                        if len(global_ids.keys()):
                                            global_ids[new_node] = max(
                                                global_ids.values()) + 1
                                        else:
                                            global_ids[new_node] = 0

                            else:
                                # TODO: overlapping
                                for node in new_nodes:
                                    if len(global_ids.keys()):
                                        global_ids[node] = max(
                                            global_ids.values()) + 1
                                    else:
                                        global_ids[node] = 0
                        else:
                            xA = np.array([global_feats[node]
                                           for node in sorted(global_feats.keys())])
                            dist = distance.squareform(distance.pdist(xA))
                            np.fill_diagonal(dist, np.inf)
                            pairs = np.argmin(dist, axis=1)
                            assert len(pairs) == len(xA)
                            pairs_dist = np.amin(dist, axis=1)
                            assert len(pairs_dist) == len(xA)
                            assigned_nodes = [
                                node for node in sorted(global_feats.keys())]
                            for i, (p, d) in enumerate(zip(pairs, pairs_dist)):
                                assert i != assigned_nodes[pairs[i]]
                                new_node = assigned_nodes[pairs[i]]
                                assigned_node = assigned_nodes[i]
                                if d <= intra_dist_thresh:
                                    for node in global_ids.keys():
                                        if global_ids[node] == global_ids[new_node]:
                                            global_ids[node] = global_ids[assigned_node]
                                    global_ids[new_node] = global_ids[assigned_node]

                        for i, video in enumerate(sorted(feats.keys())):
                            if time_step in feats[video].keys():
                                for _id in feats[video][time_step].keys():
                                    latest = sorted(map(int, [i.split(
                                        "_")[-1] for i in global_ids.keys() if "_".join((video, _id)) in i]))[-1]
                                    fos[i].write(",".join((str(time_step), str(_id), ",".join(list(map(str, bboxes[video][time_step][_id]))), "1,-1,-1,-1\n")))

                    for i, video in enumerate(sorted(feats.keys())):
                        for time_step in sorted(feats[video].keys()):
                            for _id in sorted(feats[video][time_step].keys()):
                                latest = sorted(map(int, [i.split(
                                    "_")[-1] for i in global_ids.keys() if "_".join((video, _id)) in i]))[-1]
                                merged.write(",".join((str(time_step + times[i]), str(global_ids["_".join(
                                    (video, _id, str(latest)))]), ",".join(list(map(str, bboxes[video][time_step][_id]))), "1,-1,-1,-1\n")))

                        gtfiles.append(os.path.join(
                            args.root, a, b, "gt_" + video[:-3] + "txt"))
                        tsfiles.append(os.path.join(args.root, a, b,
                                                    "pred_" + video[:-3] + "txt"))

                    for fo in fos:
                        fo.close()
                    merged.close()
                    gt = OrderedDict([(Path(f).parts[-1].split("_")[-1],
                                       mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
                    ts = OrderedDict([(Path(f).parts[-1].split("_")[-1],
                                       mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles])

                    mh = mm.metrics.create()
                    accs, names = compare_dataframes(gt, ts)

                    metrics = list(mm.metrics.motchallenge_metrics)
                    summary = mh.compute_many(
                        accs, names=names, metrics=metrics, generate_overall=True)

                    gtfiles = [(os.path.join(args.root, a, b, "gt_" +
                                             b + "-merged.txt"))]
                    tsfiles = [(os.path.join(args.root, a, b, "pred_" +
                                             b + "-merged.txt"))]

                    gt = OrderedDict([("MCMT",
                                       mm.io.loadtxt(f, fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
                    ts = OrderedDict([("MCMT",
                                       mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles])

                    accs, names = compare_dataframes(gt, ts)

                    summary = pd.concat([summary] + [mh.compute_many(
                        accs, names=names, metrics=metrics)])
                    print(mm.io.render_summary(summary, formatters=mh.formatters,
                                               namemap=mm.io.motchallenge_metric_names))
                    print()


if __name__ == "__main__":
    main(args)

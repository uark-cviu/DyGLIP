import os
import numpy as np

import argparse
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path

from numpy.lib.function_base import append
import pandas as pd
import motmetrics as mm
from motmetrics.apps.eval_motchallenge import compare_dataframes
# result_path = "../output.npz"
# input_path = "../tracklets.txt"
# output_path = "../track3.txt"
# result = np.load(result_path)
# dis_thre = 12
# dis_remove = 100
# max_length = 20


def calc_reid(result, dis_remove=100, dis_thre=12):
    distmat = result["distmat"]
    q_pids = result["q_pids"]
    g_pids = result["g_pids"]
    q_camids = result["q_camids"]
    g_camids = result["g_camids"]
    new_id = np.max(g_pids)
    # print(np.max(g_pids))
    # print(np.max(q_pids))
    rm_dict = {}
    reid_dict = {}
    indices = np.argsort(distmat, axis=1)
    num_q, num_g = distmat.shape
    # print(np.min(distmat))
    for q_idx in range(num_q):
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (
            distmat[q_idx][order] > dis_thre)
        keep = np.invert(remove)

        remove_hard = (g_pids[order] == q_pid) | (g_camids[order] == q_camid) | (
            distmat[q_idx][order] > dis_remove)
        keep_hard = np.invert(remove_hard)
        if True not in keep_hard:
            if q_camid not in list(rm_dict.keys()):
                rm_dict[q_camid] = {}
            rm_dict[q_camid][q_pid] = True
        sel_g_dis = distmat[q_idx][order][keep]
        sel_g_pids = g_pids[order][keep]
        sel_g_camids = g_camids[order][keep]
        sel_g_pids_list = []
        sel_g_camids_list = []
        selg_dis_list = []
        for i in range(sel_g_pids.shape[0]):
            sel_pid = sel_g_pids[i]
            sel_cam = sel_g_camids[i]
            sel_dis = sel_g_dis[i]
            if sel_cam not in sel_g_camids_list and sel_cam != q_camid:
                sel_g_pids_list.append(sel_pid)
                sel_g_camids_list.append(sel_cam)
                selg_dis_list.append(sel_dis)

        if len(selg_dis_list) > 0:
            new_id += 1
            if q_camid in list(reid_dict.keys()):
                if q_pid in list(reid_dict[q_camid]):
                    if reid_dict[q_camid][q_pid]["dis"] > min(selg_dis_list):
                        reid_dict[q_camid][q_pid]["dis"] = min(selg_dis_list)
                        reid_dict[q_camid][q_pid]["id"] = new_id
                else:
                    reid_dict[q_camid][q_pid] = {
                        "dis": min(selg_dis_list), "id": new_id}
            else:
                reid_dict[q_camid] = {}
                reid_dict[q_camid][q_pid] = {
                    "dis": min(selg_dis_list), "id": new_id}

        for i in range(len(sel_g_pids_list)):
            if sel_g_camids_list[i] in list(reid_dict.keys()):
                if sel_g_pids_list[i] in list(reid_dict[sel_g_camids_list[i]]):
                    if reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]]["dis"] > selg_dis_list[i]:
                        reid_dict[sel_g_camids_list[i]
                                  ][sel_g_pids_list[i]]["dis"] = selg_dis_list[i]
                        reid_dict[sel_g_camids_list[i]
                                  ][sel_g_pids_list[i]]["id"] = new_id
                else:
                    reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {
                        "dis": selg_dis_list[i], "id": new_id}
            else:
                reid_dict[sel_g_camids_list[i]] = {}
                reid_dict[sel_g_camids_list[i]][sel_g_pids_list[i]] = {
                    "dis": selg_dis_list[i], "id": new_id}

    return reid_dict, rm_dict


def calc_length(output):
    calc_dict = {}
    for line in output:
        line = line.strip().split(" ")
        cam_id = int(line[0])
        track_id = int(line[1])
        if cam_id not in list(calc_dict.keys()):
            calc_dict[cam_id] = {}
        if track_id not in list(calc_dict[cam_id].keys()):
            calc_dict[cam_id][track_id] = 1
        else:
            calc_dict[cam_id][track_id] += 1
    return calc_dict


def update_output(data, reid_dict, rm_dict, merge_mc=True):
    gtfiles = []
    tsfiles = []
    length_dict = {("MCT", "Dataset1"): 24000, ("MCT", "Dataset2"): 24000, ("MCT", "Dataset3"): 5251, ("MCT", "Dataset4"): 36001,
    (".", "PETS09"): 795, ("EPFL", "Laboratory"): 2955, ("EPFL", "Basketball"): 9368, ("EPFL", "Passageway"): 2500, ("EPFL", "Terrace"): 5010,
    ("EPFL", "Campus"): 5884, ("CAMPUS", "Parkinglot"): 6478, ("CAMPUS", "Auditorium"): 5458, ("CAMPUS", "Garden1"): 2983, ("CAMPUS", "Garden2"): 6000}

    cur_frid = 0

    if merge_mc:
        cur_frame = 0
        merged_pred_fo = open(os.path.join(
            data.datapath, "pred_" + data.datapath.split("/")[-1] + "-merged.txt"), "w")

    for idx, cam in enumerate(sorted(data.pickle.keys())):
        gtfiles.append(os.path.join(data.datapath, "gt_" + cam[:-3] + "txt"))
        tsfiles.append(os.path.join(data.datapath, "pred_" + cam[:-3] + "txt"))

        with open(os.path.join(data.datapath, "pred_" + cam[:-3] + "txt"), "w") as f:
            for line in data.testloader_dict['test']:
                frame = line[3]
                cam_id = line[2]
                if cam != cam_id:
                    continue
                track_id = line[1]
                pred_id = str(track_id)
                if cam_id in list(rm_dict.keys()):
                    if track_id in list(rm_dict[cam_id].keys()):
                        continue
                if cam_id in list(reid_dict.keys()):
                    if track_id in list(reid_dict[cam_id].keys()):
                        pred_id = str(reid_dict[cam_id][track_id]["id"])
                f.write(
                    ",".join((str(line[3]), pred_id, line[4], "1,-1,-1,-1\n")))
                if merge_mc:
                    merged_pred_fo.write(",".join(
                        (str(frame + cur_frid), pred_id, line[4], "1,-1,-1,-1\n")))
        cur_frid = length_dict[(data.datapath.split("/")[-2], data.datapath.split("/")[-1])]*(idx + 1)

    if merge_mc:
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

    if merge_mc:
        gtfiles = [(os.path.join(data.datapath, "gt_" +
                                 data.datapath.split("/")[-1] + "-merged.txt"))]
        tsfiles = [(os.path.join(data.datapath, "pred_" +
                                 data.datapath.split("/")[-1] + "-merged.txt"))]

        gt = OrderedDict([("MCMT", mm.io.loadtxt(f,
                                                 fmt="mot15-2D", min_confidence=1)) for f in gtfiles])
        ts = OrderedDict([("MCMT",
                           mm.io.loadtxt(f, fmt="mot15-2D")) for f in tsfiles])

        accs, names = compare_dataframes(gt, ts)

        summary = pd.concat([summary] + [mh.compute_many(
            accs, names=names, metrics=metrics)])
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))

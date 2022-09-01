
import os
import sys
import time
import datetime
import os.path as osp
import numpy as np
import argparse
import pickle
import cv2

def extract_im_api(args):
    base_path = args[0]
    data_path = args[1]
    split = args[2]
    cam = args[3]
    extrac_im(base_path, data_path, split, cam)


def sort_tracklets(gts):
    sorted_gts = {}
    for frame in sorted(gts.keys()):
        for car_id in gts[frame].keys():
            # line: track_id xmin ymin xmax ymax frame_number lost occluded generated label
            # frame = int(line[1])
            left = gts[frame][car_id][0]
            top = gts[frame][car_id][1]
            right = gts[frame][car_id][2]
            bot = gts[frame][car_id][3]
            # if line[9] == '"PERSON"':
            #     if int(line[8]):
            #         if int(line[6]) or int(line[7]):
            #             continue
            #     elif int(line[6]):
            #         continue
            if frame not in list(sorted_gts.keys()):
                sorted_gts[frame] = []
            sorted_gts[frame].append([left, top, right, bot, int(car_id)])
    return sorted_gts

def extrac_im(base_path, data_path, split, cam):
    print("start cam:", cam)
    # bboxes[cam] = {}
    cam_dir = os.path.join(base_path, split, cam)
    bboxes = pickle.load(open(os.path.join(data_path, split, "bboxes.pkl"), "rb"))
    cap = cv2.VideoCapture(os.path.join(cam_dir))
    sorted_gts = sort_tracklets(bboxes[cam])
    fr_id = 0
    state, im = cap.read()
    frames = list(sorted_gts.keys())
    out = cv2.VideoWriter(os.path.join(base_path, split, "visualized_" + cam),
                            cv2.VideoWriter_fourcc('M', 'J', 'P', 'G') if cam[-3:] == "avi" else cv2.VideoWriter_fourcc(*'mp4v'), 25, (im.shape[1], im.shape[0]))

    # if not os.path.exists(os.path.join(data_path, split, cam)):
    #     shutil.rmtree(os.path.join(data_path, split, cam))
        # os.makedirs(os.path.join(data_path, split, cam))
    while(state):
        print(fr_id, end="\r")
        if fr_id not in frames or im is None:
            out.write(im)
            state, im = cap.read()
            fr_id += 1
        else:
            # bboxes[cam][fr_id] = {}
            tracks = sorted_gts[fr_id]
            for track in tracks:
                left, top, right, bot, car_id = track
                cv2.rectangle(im, (left, top), (right, bot), (255, 0, 0), 2)
            out.write(im)
            state, im = cap.read()
            fr_id += 1

    out.release()


def main(args):
    video_path = args.video_path
    pred_path = args.pred_path
    splits = os.listdir(pred_path)

    video = [("CAMPUS", "Auditorium")]
    # video = [("EPFL", "Basketball"), ("EPFL", "Campus"), ("EPFL", "Laboratory"), ("EPFL", "Passageway"), ("EPFL", "Terrace"), (".", "PETS09")]
    # video = [("MCT", "Dataset1"), ("MCT", "Dataset2"), ("MCT", "Dataset3"), ("MCT", "Dataset4")]
    for data, split in video:
        args_list = []
        split_dir = os.path.join(video_path, data, split)
        cams = [i for i in os.listdir(split_dir) if (
            ".mp4" in i or ".avi" in i)]
        for cam in cams:
            camid = cam.split("/")[-1]
            args_list.append(
                [os.path.join(video_path, data), os.path.join(pred_path, data), split, camid])
        for arg in args_list:
            extract_im_api(arg)

def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video_path', type=str, default="",
                        help='path to the aicity 2020 track 3 folders')
    parser.add_argument('--pred_path', type=str, default="./exp",
                        help='path to the output dictionaries')
    return parser

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)


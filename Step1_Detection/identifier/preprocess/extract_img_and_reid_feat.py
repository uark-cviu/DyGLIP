import os
import pickle
import cv2
from multiprocessing import Pool
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from torchreid import models, utils
from torchvision import transforms
from PIL import Image

import GPUtil
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
model = models.osnet_x1_0(702)
state_dict = torch.load(
    "osnet_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth")
model.load_state_dict(state_dict)
model.eval()
model.to(device)
computed_feats = {}



def argument_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data_path', type=str, default="",
                        help='path to the aicity 2020 track 3 folders')
    parser.add_argument('--output_path', type=str, default="./exp/imgs/",
                        help='path to the output dictionaries')
    parser.add_argument("--njobs", type=int, default=4,
                        help="number of pools to extract imgs")
    return parser


def preprocess(images):
    normalizer = transforms.Compose([
        transforms.Resize((256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    image = normalizer(images)
    return image


def sort_tracklets(preds):
    sorted_preds = {}
    for frame in sorted(preds.keys()):
        for car_id in preds[frame].keys():
            # line: track_id xmin ymin xmax ymax frame_number lost occluded generated label
            # frame = int(line[1])
            left = preds[frame][car_id][0]
            top = preds[frame][car_id][1]
            right = preds[frame][car_id][2] + left
            bot = preds[frame][car_id][3] + top
            # if line[9] == '"PERSON"':
            #     if int(line[8]):
            #         if int(line[6]) or int(line[7]):
            #             continue
            #     elif int(line[6]):
            #         continue
            if frame not in list(sorted_preds.keys()):
                sorted_preds[frame] = []
            sorted_preds[frame].append([left, top, right, bot, int(car_id)])
    return sorted_preds


def extract_im_api(args):
    base_path = args[0]
    data_path = args[1]
    split = args[2]
    cam = args[3]
    extrac_im(base_path, data_path, split, cam)


def extrac_im(base_path, data_path, split, cam):
    print("start cam:", cam)
    computed_feats[cam] = {}
    # bboxes[cam] = {}
    cam_dir = os.path.join(base_path, split, cam)
    bboxes = pickle.load(open(os.path.join(data_path, split, "bboxes.pkl"), "rb"))
    cap = cv2.VideoCapture(os.path.join(cam_dir))
    sorted_preds = sort_tracklets(bboxes[cam])
    fr_id = 0
    state, im = cap.read()
    frames = list(sorted_preds.keys())
    # if not os.path.exists(os.path.join(data_path, split, cam)):
    #     shutil.rmtree(os.path.join(data_path, split, cam))
        # os.makedirs(os.path.join(data_path, split, cam))
    while(state):
        print(fr_id, end="\r")
        if fr_id not in frames or im is None:
            state, im = cap.read()
            fr_id += 1
        else:
            computed_feats[cam][fr_id] = {}
            # bboxes[cam][fr_id] = {}
            tracks = sorted_preds[fr_id]
            for track in tracks:
                left, top, right, bot, car_id = track

                clip = im[top:bot, left:right]
                if int(top) <= 0 or int(left) <= 0 or int(bot-top) <= 1 or int(right-left) <= 1: continue
                car_num = str(car_id).zfill(5)
                im_name = car_num+"_"+cam+"_"+str(fr_id).zfill(4)+".jpg"
                if not os.path.exists(os.path.join(data_path, split, cam, car_num)):
                    os.makedirs(os.path.join(data_path, split, cam, car_num))
                cv2.imwrite(os.path.join(data_path, split, cam, car_num, im_name), clip)

                clip = cv2.cvtColor(clip, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(clip)
                processed_img = preprocess(im_pil).to(device)
                with torch.no_grad():
                    features = model(processed_img.unsqueeze(0))

                features = features.cpu().detach()
                computed_feats[cam][fr_id][car_num] = features
                bboxes[cam][fr_id][car_num] = [left, top, right, bot]
            state, im = cap.read()
            fr_id += 1


def main(args):
    base_path = args.data_path
    data_path = args.output_path
    splits = os.listdir(base_path)
    available_gpus = GPUtil.getAvailable()
    n_available_gpu = len(available_gpus)

    # video = [("CAMPUS", "Auditorium"), ("CAMPUS", "Garden1"), ("CAMPUS", "Garden2"), ("CAMPUS", "Parkinglot")]
    video = [("EPFL", "Basketball"), ("EPFL", "Campus"), ("EPFL", "Laboratory"), ("EPFL", "Passageway"), ("EPFL", "Terrace"), (".", "PETS09")]
    # video = [("aic", "S05")]
    for data, split in video:
        global computed_feats
        computed_feats = {}
        # global bboxes
        # bboxes = {}
        args_list = []
        split_dir = os.path.join(base_path, data, split)
        cams = [i for i in os.listdir(split_dir) if (
            ".mp4" in i or ".avi" in i)]
        for cam in cams:
            camid = cam.split("/")[-1]
            args_list.append(
                [os.path.join(base_path, data), os.path.join(data_path, data), split, camid])
        for arg in args_list:
            extract_im_api(arg)
        with open(os.path.join(data_path, data, split, "reid_feats.pkl"), "wb") as f:
            pickle.dump(computed_feats, f)
        # with open(os.path.join(data_path, split, "bboxes.pkl"), "wb") as f:
        #     pickle.dump(bboxes, f)
    # n_jobs = args.njobs
    # pool = Pool(n_jobs)
    # pool.map(extract_im_api, args_list)
    # pool.close()



if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

import os
import pickle
import cv2
from multiprocessing import Pool
import argparse
import shutil
import numpy as np
import torch
import torch.nn.functional as F
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances

import GPUtil
import torch

CFG_FILES = {
    'res50': 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml',
    'res101': 'COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml',
    'res101x': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml',
}

DEFAULT_MODEL = 'res101'
cfg_file = CFG_FILES[DEFAULT_MODEL]
device = 'cuda:0' if torch.cuda.is_available() else "cpu"
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(cfg_file))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_file)
cfg.MODEL.DEVICE = device
predictor = DefaultPredictor(cfg)
computed_feats = {}
bboxes = {}

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
    processed_images = []
    for image in images:
        height, width = image.shape[:2]
        image = image.to(device=device, non_blocking=True)
        image = image.permute(2, 0, 1).type(torch.float)
        origin_ratio = width / height
        cfg_ratio = cfg.INPUT.MAX_SIZE_TEST / cfg.INPUT.MIN_SIZE_TEST
        if cfg_ratio > origin_ratio:
            target_height = cfg.INPUT.MIN_SIZE_TEST
            target_width = int(round(target_height * origin_ratio))
        else:
            target_width = cfg.INPUT.MAX_SIZE_TEST
            target_height = int(round(target_width / origin_ratio))
        target_shape = (target_height, target_width)
        image = F.interpolate(image.unsqueeze(0), target_shape,
                              mode='bilinear', align_corners=False)
        image = (image.squeeze(0) - predictor.model.pixel_mean) / \
            predictor.model.pixel_std
        processed_images.append(image)
    images = ImageList.from_tensors(
        processed_images, predictor.model.backbone.size_divisibility)
    return images

def sort_tracklets(_preds):
    with open(_preds[:-3] + "txt", "r") as f:
        preds = f.readlines()
    sorted_preds = {}
    car_list = []
    for line in preds:
        line = line.strip().split(",")
        # line: track_id xmin ymin xmax ymax frame_number lost occluded generated label
        frame = int(line[0])
        left = int(line[2])
        top = int(line[3])
        right = int(line[4]) + int(line[2])
        bot = int(line[5]) + int(line[3])
        car_id = int(line[1])
        # if line[9] == '"PERSON"':
        #     if int(line[6]):
        #         continue
        if frame not in list(sorted_preds.keys()):
            sorted_preds[frame] = []
        sorted_preds[frame].append([left, top, right, bot, car_id])
    return sorted_preds


def extract_im_api(args):
    base_path = args[0]
    data_path = args[1]
    split = args[2]
    cam = args[3]
    extrac_im(base_path, data_path, split, cam)


def extrac_im(base_path, data_path, split, cam):
    print("start cam:"+cam)
    computed_feats[cam] = {}
    bboxes[cam] = {}
    cam_dir = os.path.join(base_path, split, cam)
    cap = cv2.VideoCapture(os.path.join(cam_dir))
    sorted_preds = sort_tracklets(cam_dir)
    fr_id = 0
    state, im = cap.read()
    frames = list(sorted_preds.keys())
    if not os.path.exists(os.path.join(data_path, split, cam)):
    #     shutil.rmtree(os.path.join(data_path, split, cam))
        os.makedirs(os.path.join(data_path, split, cam))
    while(state):
        if fr_id not in frames or im is None:
            state, im = cap.read()
            fr_id += 1
        else:
            computed_feats[cam][fr_id] = {}
            bboxes[cam][fr_id] = {}
            tracks = sorted_preds[fr_id]
            for track in tracks:
                left, top, right, bot, car_id = track

                clip = im[top:bot, left:right]
                car_num = str(car_id).zfill(5)
                im_name = car_num+"_"+cam+"_"+str(fr_id).zfill(4)+".jpg"
                if not os.path.exists(os.path.join(data_path, split, cam, car_num)):
                    os.makedirs(os.path.join(data_path, split, cam, car_num))
                cv2.imwrite(os.path.join(data_path, split,
                                         cam, car_num, im_name), clip)

                clip = cv2.cvtColor(clip, cv2.COLOR_BGR2RGB)
                bb = [0, 0, clip.shape[1], clip.shape[0]]
                processed_img = preprocess([torch.as_tensor(clip)])
                features = predictor.model.backbone(processed_img.tensor)
                # reference: https://github.com/airsplay/py-bottom-up-attention/blob/master/demo/demo_feature_extraction_given_box.ipynb
                features = [features[f]
                            for f in predictor.model.roi_heads.in_features]

                raw_boxes = Boxes(torch.from_numpy(np.array([bb])).cuda())
                raw_boxes.scale(
                    scale_x=1.0*processed_img.tensor.shape[-1]/clip.shape[1], scale_y=1.0*processed_img.tensor.shape[-2]/clip.shape[0])
                box_features = predictor.model.roi_heads.mask_pooler(features, [
                                                                     raw_boxes])
                features = box_features.mean(dim=(2, 3))
                features = features / features.norm(dim=1, keepdim=True)
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

    # args_list = []
    for split in splits:
        global computed_feats
        computed_feats = {}
        global bboxes
        bboxes = {}
        args_list = []
        split_dir = os.path.join(base_path, split)
        cams = [i for i in os.listdir(split_dir) if (
            ".mp4" in i or ".avi" in i) and (i[:-3] + "txt" in os.listdir(split_dir))]
        for cam in cams:
            camid = cam.split("/")[-1]
            args_list.append(
                [base_path, data_path, split, camid])
        for arg in args_list:
            extract_im_api(arg)
        with open(os.path.join(data_path, split, "maskrcnn_feats.pkl"), "wb") as f:
            pickle.dump(computed_feats, f)
        with open(os.path.join(data_path, split, "bboxes.pkl"), "wb") as f:
            pickle.dump(bboxes, f)
    # n_jobs = args.njobs
    # pool = Pool(n_jobs)
    # pool.map(extract_im_api, args_list)
    # pool.close()



if __name__ == "__main__":
    parser = argument_parser()
    args = parser.parse_args()
    main(args)

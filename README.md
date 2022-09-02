# DyGLIP: A dynamic graph model with link prediction for accurate multi-camera multiple object tracking

Authors: Kha Gia Quach, Pha Nguyen, Huu Le, Thanh-Dat Truong, Chi Nhan Duong, Minh-Triet Tran, Khoa Luu

Email: kquach@ieee.org, panguyen@uark.edu

## Overview
We release the code for our paper in CVPR 2021.
For more information please refer to our accepted paper in CVPR 2021.

## Project Download

Firstly please download the project through:
```
git clone https://github.com/uark-cviu/DyGLIP
```

## Prerequisites
The code requires the following libraries to be installed:
- [Python] (https://python.org/) 3.6+
- [PyTorch](https://pytorch.org/) >= 1.5
- [CUDA] (https://developer.nvidia.com/cuda-downloads) 9.0+

```
cd DyGLIP
conda env create -f environment.yml 
conda activate dyglip
```

## Data Preparation

Please place all datasets in `/data/`:

```
/data/
├── CAMPUS
│   ├── Auditorium
│   ├── Garden1
│   ├── Garden2
│   └── Parkinglot
├── EPFL
│   ├── Basketball
│   ├── Campus
│   ├── Laboratory
│   ├── Passageway
│   └── Terrace
├── PETS09
├── MCT
│   ├── Dataset1
│   ├── Dataset2
│   ├── Dataset3
│   └── Dataset4
└── aic
    ├── S02
    └── S05
```

### Step 1 - Detection

Please follow [detection guidance](./Step1_Detection/README.md) to get bounding boxes prediction.

Get maskrcnn features by running the file `Step1_Detection/identifier/preprocess/extract_img_and_feat.py`, the output should be two files: `bboxes.pkl` and `maskrcnn_feats.pkl`.

Get pre-computed reid features by running the file `Step1_Detection/identifier/preprocess/extract_img_and_reid_feat.py`, the output should be the `reid_feats.pkl` file.

### Step 2 - Graph-based Feature Extraction

This [environment](Step2_GraphFeature/requirements.txt) must be python 2.7, tensorflow 1.11

Prepare graph `Step2_GraphFeature/prepare_graphs.py`

Please follow [GraphFeature guidance](./Step2_GraphFeature/README.md) to train the model.

### Step 3 - Matching

Get output from matching baselines [graph](./Step3_Matching/graph_baseline.py) and [non-negative matrix factorization](./Step3_Matching/nmf.py).

## Acknowledgements

- Thanks [DySAT](https://github.com/aravindsankar28/DySAT) for providing strong baseline for graph attention network.
- Thanks [ELECTRICITY-MTMC](https://github.com/KevinQian97/ELECTRICITY-MTMC) for providing useful detection inference pipeline for MC-MOT.

## Citation

If you find this code useful for your research, please consider citing:

```bib
@InProceedings{Quach_2021_CVPR,
    author    = {Quach, Kha Gia and Nguyen, Pha and Le, Huu and Truong, Thanh-Dat and Duong, Chi Nhan and Tran, Minh-Triet and Luu, Khoa},
    title     = {DyGLIP: A Dynamic Graph Model With Link Prediction for Accurate Multi-Camera Multiple Object Tracking},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {13784-13793}
}
```
[Paper Linkage](https://openaccess.thecvf.com/content/CVPR2021/html/Quach_DyGLIP_A_Dynamic_Graph_Model_With_Link_Prediction_for_Accurate_CVPR_2021_paper.html)

```bib
@InProceedings{Nguyen_2022_CVPR,
    author    = {Nguyen, Pha and Quach, Kha Gia and Duong, Chi Nhan and Le, Ngan and Nguyen, Xuan-Bac and Luu, Khoa},
    title     = {Multi-Camera Multiple 3D Object Tracking on the Move for Autonomous Vehicles},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2022},
    pages     = {2569-2578}
}
```

[Paper Linkage](https://openaccess.thecvf.com/content/CVPR2022W/Precognition/html/Nguyen_Multi-Camera_Multiple_3D_Object_Tracking_on_the_Move_for_Autonomous_CVPRW_2022_paper.html)

from __future__ import division
from __future__ import print_function
from sys import version_info
from collections import OrderedDict
from motmetrics.apps.eval_motchallenge import compare_dataframes
import motmetrics as mm
import pandas as pd
from numpy.lib.function_base import append
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy import sparse
from utils.utilities import *
from utils.preprocess import *
from utils.minibatch import *
from models.DyGLIP.models import DyGLIP
from flags import *
from eval.link_prediction import evaluate_classifier, evaluate_classifier_list, write_to_csv, predict_link
import os.path as osp
import scipy
import logging
from datetime import datetime
import time
import os
import json
import warnings
warnings.filterwarnings("ignore")


if version_info[0] < 3:
    import cPickle
else:
    import pickle as cPickle

np.random.seed(123)
tf.set_random_seed(123)

flags = tf.app.flags
FLAGS = flags.FLAGS

length_dict = {("MCT", "Dataset1"): 24000, ("MCT", "Dataset2"): 24000, ("MCT", "Dataset3"): 5251, ("MCT", "Dataset4"): 36001,
               (".", "PETS09"): 795, ("EPFL", "Laboratory"): 2955, ("EPFL", "Basketball"): 9368, ("EPFL", "Passageway"): 2500, ("EPFL", "Terrace"): 5010,
               ("EPFL", "Campus"): 5884, ("CAMPUS", "Parkinglot"): 6478, ("CAMPUS", "Auditorium"): 5458, ("CAMPUS", "Garden1"): 2983, ("CAMPUS", "Garden2"): 6000}

a, b = FLAGS.dataset_path.split("/")[-2], FLAGS.dataset_path.split("/")[-1]


# FLAGS.dataset_path = osp.join(test_root, a, b)
# FLAGS.dataset = "_".join((a, b))
# Assumes a saved base model as input and model name to get the right directory.
output_dir = "./logs/{}_{}/".format(FLAGS.base_model, FLAGS.model)
dataset_cache_dir = "{}/{}/".format(
    FLAGS.dataset_cache_path, FLAGS.dataset)

if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

if not os.path.isdir(dataset_cache_dir):
    os.mkdir(dataset_cache_dir)

config_file = output_dir + "flags_{}.json".format(FLAGS.dataset)

# with open(config_file, 'r') as f:
#     config = json.load(f)
#     for name, value in config.items():
#         if name in FLAGS.__flags:
#             FLAGS.__flags[name].value = value

print("Updated flags", FLAGS.flag_values_dict().items())

# Set paths of sub-directories.
LOG_DIR = output_dir + FLAGS.log_output_dir
SAVE_DIR = output_dir + FLAGS.save_dir
CSV_DIR = output_dir + FLAGS.csv_dir
# MODEL_DIR = output_dir + FLAGS.model_dir
# BEST_MODEL_DIR = output_dir + FLAGS.best_model_dir
MODEL_DIR = "best_model"
BEST_MODEL_DIR = "best_model"
OUTPUT_MOT_DIR = output_dir + FLAGS.output_mot_dir

if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)

if not os.path.isdir(CSV_DIR):
    os.mkdir(CSV_DIR)

if not os.path.isdir(MODEL_DIR):
    os.mkdir(MODEL_DIR)

if not os.path.isdir(BEST_MODEL_DIR):
    os.mkdir(BEST_MODEL_DIR)

if not os.path.isdir(OUTPUT_MOT_DIR):
    os.mkdir(OUTPUT_MOT_DIR)

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.GPU_ID)

datetime_str = datetime.now().strftime("%Y%m%d_%H%M%S")
today = datetime.today()

# Setup logging
log_file = LOG_DIR + '/%s_%s_%s_%s_%s.log' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day), str(FLAGS.time_steps))

log_level = logging.INFO
logging.basicConfig(filename=log_file, level=log_level, format='%(asctime)s - %(levelname)s: %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S')

logging.info(FLAGS.flag_values_dict().items())

# Create file name for result log csv from certain flag parameters.
output_file = CSV_DIR + '/%s_%s_%s_%s.csv' % (FLAGS.dataset.split("/")[0], str(today.year),
                                              str(today.month), str(today.day))

# model_dir is not used in this code for saving.

# utils folder: utils.py, random_walk.py, minibatch.py
# models folder: layers.py, models.py
# main folder: train.py
# eval folder: link_prediction.py

"""
#1: Train logging format: Create a new log directory for each run (if log_dir is provided as input).
Inside it,  a file named <>.log will be created for each time step. The default name of the directory is "log" and the
contents of the <>.log will get appended per day => one log file per day.

#2: Model save format: The model is saved inside model_dir.

#3: Output save format: Create a new output directory for each run (if save_dir name is provided) with embeddings at
each
time step. By default, a directory named "output" is created.

#4: Result logging format: A csv file will be created at csv_dir and the contents of the file will get over-written
as per each day => new log file for each day.
"""

file_name = "{}/{}/graph_cache.npz".format(
    FLAGS.dataset_cache_path, FLAGS.dataset)
mapping_filename = "{}/{}/graph_mapping_cache.pkl".format(
    FLAGS.dataset_cache_path, FLAGS.dataset)

if not os.path.exists(file_name) or not os.path.exists(mapping_filename):
    graphs, adjs, mapping = load_graphs(FLAGS.dataset_path)

    np.savez(file_name, graphs=graphs, adjs=adjs)

    cPickle.dump(mapping, open(mapping_filename, 'wb'))
else:
    data = np.load(file_name, allow_pickle=True)

    graphs = data['graphs']
    adjs = data['adjs']
    #mapping = data['mapping']
    mapping = cPickle.load(open(mapping_filename, 'rb'))

    print("Loaded {} graphs from cached".format(len(graphs)))

num_time_steps = len(graphs)

# if FLAGS.featureless:
#     feats = [scipy.sparse.identity(adjs[num_time_steps - 1].shape[0]).todense()[range(0, x.shape[0]), :] for x in adjs if
#              x.shape[0] <= adjs[num_time_steps - 1].shape[0]]
# else:
frames = set()
file_name = "{}/{}/feature_cache_full.pkl".format(
    FLAGS.dataset_cache_path, FLAGS.dataset)

if not os.path.exists(file_name):
    _feats = load_feats(FLAGS.dataset_path)
    feats = []
    latest_feats = {}
    count_t = 0

    for video in sorted(_feats.keys()):
        frames.update(_feats[video].keys())
    frames = sorted(frames)

    for x in adjs[:num_time_steps]:

        g_f = []

        graph_idx = count_t
        # index[count_t]     # if not sink graph_idx_org = graph_idx
        graph_idx_org = graph_idx
        f_idx = frames[graph_idx_org]

        # print('No. of nodes = {} at time {} - {}'.format(
        #     len(graphs[graph_idx].nodes()), count_t, f_idx))

        node_list = list(graphs[graph_idx].nodes())
        # node_list_org = list(graphs_org[graph_idx].nodes())
        for node_idx, node in enumerate(node_list):
            name = list(mapping.keys())[list(mapping.values()).index(node)]
            name_org = name  # node_list_org[node_idx]
            video = "_".join(name.split("_")[0:-2])
            track = name.split("_")[-2]

            name = video + '_' + track

            if f_idx in _feats[video].keys() and track in _feats[video][f_idx].keys():
                g_f.append(_feats[video][f_idx][track].ravel())
                latest_feats[name] = _feats[video][f_idx][track].ravel()
                # print('Add new feature for node {} - {}'.format(name, name_org))
            elif name in latest_feats.keys():
                g_f.append(latest_feats[name])
                # print(
                #     'Add existing feature for node {} - {}'.format(name, name_org))
            else:
                found = False
                for _t in range(f_idx, 0, -1):
                    print("Loading time_step ", f_idx,
                          " go back ", _t, end="\r")
                    if _t in _feats[video].keys() and track in _feats[video][_t].keys():
                        g_f.append(_feats[video][_t][track].ravel())
                        print(
                            'Add feature for node {} - {}'.format(name, name_org))
                        found = True
                        break
                if not found:
                    print('Not found node {} - {}'.format(name, name_org))
                    exit(0)

        assert adjs[count_t].shape[0] == len(g_f)

        if len(g_f) > 0:
            feats.append(np.array(g_f))

        count_t += 1
        print("Loading time_step ", count_t, end="\r")

    cPickle.dump(feats, open(file_name, 'wb'))

else:
    _feats = load_feats(FLAGS.dataset_path)

    for video in sorted(_feats.keys()):
        frames.update(_feats[video].keys())
    frames = sorted(frames)

    feats = cPickle.load(open(file_name, 'rb'))

    print("Loaded {} features from cached".format(feats[0].shape[1]))

num_features = feats[0].shape[1]
assert num_time_steps < len(adjs) + 1  # So that, (t+1) can be predicted.

adj_train = []
feats_train = []
num_features_nonzero = []
loaded_pairs = False

for i in range(0, num_time_steps - FLAGS.window + 1, FLAGS.window):

    idx = i + FLAGS.window - 1  # (i + 1) * FLAGS.window - 1

    new_G = nx.MultiGraph()
    new_G.add_nodes_from(graphs[idx].nodes(data=True))

    if not os.path.isdir(LOG_DIR + '/graph'):
        os.mkdir(LOG_DIR + '/graph')

    # file_name = os.path.join(LOG_DIR, 'graph', 'graphvis_{}_t-1_GT.png'.format(i))
    #
    # A_before = to_agraph(graphs[idx])
    # A_before.layout('dot')
    # A_before.draw(file_name)

    for e in graphs[idx - 1].edges():
        new_G.add_edge(e[0], e[1])

    ## predicting edge using raw feature
    # get new nodes to pair
    pairs = []
    for e in graphs[idx].edges():
        if e not in graphs[idx - 1].edges():
            pairs.append( (e[0], e[1]))

    graphs[idx] = new_G
    adjs[idx] = nx.adjacency_matrix(new_G)

    # re-assign adjs matrix
    if len(pairs) > 0:
        pred_new = predict_link(pairs, [feats[idx]], [feats[idx]])

        for e_idx, e in enumerate(pairs):
            adjs[idx][e[0], e[1]] = pred_new[0][e_idx]
    #
    # file_name = os.path.join(LOG_DIR, 'graph', 'graphvis_{}_t-1_train.png'.format(i))
    #
    # A_after = to_agraph(graphs[idx])
    # #print(A)
    # A_after.layout('dot')
    # A_after.draw(file_name)

# Load training context pairs (or compute them if necessary)
# context_pairs_train = []
# for i in range(0, num_time_steps):
#     context_pairs_train.append(
#         run_random_walks_n2v(graphs[i], graphs[i].nodes()))

# Load evaluation data.
# train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
#     get_evaluation_data(adjs, num_time_steps, FLAGS.dataset, val_mask_fraction=0.0, test_mask_fraction=1.0)

# Create the adj_train so that it includes nodes from (t+1) but only edges from t: this is for the purpose of
# inductive testing.


# print("# train: {}, # val: {}, # test: {}".format(
#     len(train_edges), len(val_edges), len(test_edges)))
# logging.info("# train: {}, # val: {}, # test: {}".format(
#     len(train_edges), len(val_edges), len(test_edges)))

# Normalize and convert adj. to sparse tuple format (to provide as input via SparseTensor)
adj_train = list(map(lambda adj: normalize_graph_gcn(
    adj.todense(), False), adjs))      # False not convert to tupe

feats_train = list(
    map(lambda feat: preprocess_features(feat, False)[1], feats))
num_features_nonzero = [x[1].shape[0] for x in feats_train]


def construct_placeholders(num_time_steps):
    min_t = 0
    max_t = num_time_steps

    if FLAGS.window > 0:
        min_t = 0  # max(num_time_steps - FLAGS.window - 1, 0)
        max_t = FLAGS.window

    placeholders = {
        # 'node_1': [tf.placeholder(tf.int32, shape=(None,), name="node_1") for _ in range(min_t, max_t)],
        # # [None,1] for each time step.
        # 'node_2': [tf.placeholder(tf.int32, shape=(None,), name="node_2") for _ in range(min_t, max_t)],
        # [None,1] for each time step.
        # [None,1]
        'batch_nodes': tf.placeholder(tf.int32, shape=(None,), name="batch_nodes"),
        'features': [tf.placeholder(tf.float32, shape=(None, num_features), name="feats") for _ in
                     range(min_t, max_t)],
        'adjs': [tf.placeholder(tf.float32, shape=(None, None), name="adjs") for i in
                 range(min_t, max_t)],
        'spatial_drop': tf.placeholder(dtype=tf.float32, shape=(), name='spatial_drop'),
        'temporal_drop': tf.placeholder(dtype=tf.float32, shape=(), name='temporal_drop')
    }
    return placeholders


print("Initializing session")
# Initialize session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

placeholders = construct_placeholders(num_time_steps)

minibatchIterator = NodeMinibatchIterator(graphs, feats_train, adj_train,
                                          placeholders, num_time_steps, batch_size=FLAGS.batch_size)
print("# training batches per epoch",
      minibatchIterator.num_training_batches())

model = DyGLIP(placeholders, num_features,
              num_features_nonzero, minibatchIterator.degs, eval=True)
# sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(
    sess, "{}/model-{}.ckpt".format(BEST_MODEL_DIR, FLAGS.epochs))
# Result accumulator variables.
epochs_test_result = defaultdict(lambda: [])
epochs_val_result = defaultdict(lambda: [])
epochs_embeds = []
epochs_attn_wts_all = []
minibatchIterator.test_reset()
# feed_dict = minibatchIterator.batch_feed_dict(minibatchIterator.train_nodes)

raw_emb_all = []
att_emb_all = []
total_t = 0
print(len(graphs), len(minibatchIterator.train_nodes))
# max_iter = len(minibatchIterator.train_nodes)
max_iter = 795//3
import time

first = time.time()
for t in tqdm(range(max_iter)):  # len(minibatchIterator.train_nodes))):
    feed_dict = minibatchIterator.batch_feed_dict(
        minibatchIterator.train_nodes[t], minibatchIterator.train_graph_idx[t])
    feed_dict.update({placeholders['spatial_drop']: 0.0})
    feed_dict.update({placeholders['temporal_drop']: 0.0})
    if FLAGS.window < 0:
        assert FLAGS.time_steps == model.final_output_embeddings.get_shape()[
            1]
    emb, emb2 = sess.run([model.final_output_embeddings,
                          model.structural_output_embeddings], feed_dict=feed_dict)
    # List[N x List[5 x feature]] features: n_node x dimension
    raw_emb_all.append(emb2)
    # List[N x featuers] features: n_node x 5 x dimension
    att_emb_all.append(emb)

    #print(len(emb2), emb2[0].shape, emb2[1].shape, emb.shape)
    total_t += len(emb2)
    # if t == 20:
    #    break

last = time.time()
print(len(graphs), "TIME: ", last - first)
current_t = 0
state_dict = {}

for idx in tqdm(range(len(raw_emb_all))):
    time_size = len(raw_emb_all[idx])
    if idx == len(raw_emb_all) - 1:
        current_t = len(graphs) - FLAGS.window
    for dt in range(time_size):
        t = current_t + dt
        n_node = raw_emb_all[idx][dt].shape[0]
        assert n_node == len(list(graphs[t].node.keys()))
        f_idx = frames[t]
        for i in range(n_node):
            raw_feat = raw_emb_all[idx][dt][i]
            att_feat = att_emb_all[idx][i][dt]
            node = list(graphs[t].node.keys())[i]
            name = list(mapping.keys())[list(mapping.values()).index(node)]
            video = "_".join(name.split("_")[0:-2])
            track = name.split("_")[-2]
            if video not in state_dict:
                state_dict[video] = {}
            if f_idx not in state_dict[video]:
                state_dict[video][f_idx] = {}
            if track not in state_dict[video][f_idx]:
                state_dict[video][f_idx][track] = {}

            state_dict[video][f_idx][track]["raw"] = raw_feat
            state_dict[video][f_idx][track]["att"] = att_feat

    current_t += time_size

# CAMPUS-Garder
cPickle.dump(state_dict, open(os.path.join(
    OUTPUT_MOT_DIR, FLAGS.dataset + ".pkl"), 'wb'))

# new_G = nx.MultiGraph()
current_nodes = []

current_t = 0
global_ids = {}

gtfiles = []
tsfiles = []
fos = [open(osp.join(FLAGS.dataset_path, "pred_" + video[:-3] + "txt"), "w")
       for video in sorted(_feats.keys())]
bboxes = pickle.load(
    open(osp.join(FLAGS.dataset_path, "bboxes.pkl"), "rb"))

merged = open(osp.join(FLAGS.dataset_path,
                       "pred_" + b + "-merged.txt"), "w")

for idx in tqdm(range(len(raw_emb_all))):
    time_size = len(raw_emb_all[idx])
    if idx != len(raw_emb_all) - 1:
        ranges = [time_size - 1]
    else:
        ranges = [len(frames) - 1 - current_t]
    for dt in ranges:
        t = current_t + dt
        f_idx = frames[t]
        new_nodes = []
        # TODO: reset graph and initialize by the baseline here if needed
        new_nodes = [node for node in graphs[t].nodes()
                     if node not in current_nodes]
        # new_G.add_node(node)
        # NOTE: This logic is predicting link at the first node's appeareance, should we link it at its end?
        for new_node in new_nodes:
            new_name = list(mapping.keys())[list(
                mapping.values()).index(new_node)]
            new_video = "_".join(new_name.split("_")[0:-2])
            new_track = new_name.split("_")[-2]
            new_feat = state_dict[new_video][f_idx][new_track]["att"]
            max_score = 0
            assigned_node = None
            for current_node in current_nodes:
                current_name = list(mapping.keys())[list(
                    mapping.values()).index(current_node)]
                current_video = "_".join(current_name.split("_")[0:-2])
                current_track = current_name.split("_")[-2]
                current_feat = state_dict[current_video][f_idx][current_track]["att"]

                test_edges_pred = predict_link([[[0, 0]]], [[new_feat]], [
                    [current_feat]], MODEL_DIR, epoch=67)
                if (test_edges_pred[0] > max_score):
                    assigned_node = current_name
                    max_score = test_edges_pred[0]
            if (max_score > 0.5):
                global_ids[new_name] = global_ids[assigned_node]
            else:
                if len(global_ids.keys()):
                    global_ids[new_name] = max(global_ids.values()) + 1
                else:
                    global_ids[new_name] = 0
            current_nodes.append(new_node)
    if idx != len(raw_emb_all) - 1:
        ranges = range(time_size)
    else:
        ranges = range(len(frames) - 1 - current_t)
    for dt in ranges:
        time_step = frames[current_t + dt]
        for i, video in enumerate(sorted(_feats.keys())):
            if time_step in _feats[video].keys():
                for _id in _feats[video][time_step].keys():
                    latest = sorted(map(int, [i.split(
                        "_")[-1] for i in global_ids.keys() if "_".join((video, _id)) in i]))[-1]
                    fos[i].write(",".join((str(time_step), str(global_ids["_".join(
                        (video, _id, str(latest)))]), ",".join(list(map(str, bboxes[video][time_step][_id]))), "1,-1,-1,-1\n")))

    current_t += time_size

times = []
videos = sorted(_feats.keys())
for video in videos:
    times.append(0 if not len(times) else len(times)*length_dict[(a, b)])

for i, video in enumerate(sorted(_feats.keys())):
    for time_step in sorted(_feats[video].keys()):
        for _id in sorted(_feats[video][time_step].keys()):
            latest = sorted(map(int, [i.split(
                "_")[-1] for i in global_ids.keys() if "_".join((video, _id)) in i]))[-1]
            merged.write(",".join((str(time_step + times[i]), str(global_ids["_".join(
                (video, _id, str(latest)))]), ",".join(list(map(str, bboxes[video][time_step][_id]))), "1,-1,-1,-1\n")))

    gtfiles.append(os.path.join(
        FLAGS.dataset_path, "gt_" + video[:-3] + "txt"))
    tsfiles.append(os.path.join(FLAGS.dataset_path,
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

gtfiles = [(os.path.join(FLAGS.dataset_path, "gt_" + b + "-merged.txt"))]
tsfiles = [(os.path.join(FLAGS.dataset_path, "pred_" + b + "-merged.txt"))]

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

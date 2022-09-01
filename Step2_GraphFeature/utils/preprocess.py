from __future__ import print_function
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
from networkx.relabel import relabel_nodes
from utils.utilities import run_random_walks_n2v
import dill
import pickle
import os

flags = tf.app.flags
FLAGS = flags.FLAGS
np.random.seed(123)


def load_graphs(dataset_str, sink=False):
    """Load graph snapshots given the name of dataset"""
    mapping = {}
    graphs = pickle.load(open(dataset_str + "/graphs.pkl", "rb"))

    new_graphs = []
    new_idx = []

    if sink:
        for g_idx, g in enumerate(graphs):

            sorted_nodes = sorted(g.nodes()) #g.nodes() #
            for i in sorted_nodes:
                if i not in mapping.keys():
                    mapping[i] = len(mapping.keys()) #'%05d' % (len(mapping.keys()))

            if g_idx > 0:
                if list(g.nodes) == list(graphs[g_idx - 1].nodes()) and len(g.edges) == len(graphs[g_idx - 1].edges()):
                    continue

            if len(g.edges) == 0:
                continue

            new_graphs.append(g)
            new_idx.append(g_idx)

        new_graphs_rename = []
        for i in range(len(new_graphs)):
            G = nx.MultiGraph()
            tmp_G = relabel_nodes(new_graphs[i], mapping=mapping)

            G.add_nodes_from(sorted(tmp_G.nodes()))
            G.add_edges_from(sorted(tmp_G.edges()))

            new_graphs_rename.append(G)

        print("Loaded {} graphs ".format(len(new_graphs_rename)))
        adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), new_graphs_rename))
        return new_graphs_rename, new_graphs, adj_matrices, mapping, new_idx
    else:
        for g_idx, g in enumerate(graphs):
            sorted_nodes = sorted(g.nodes()) # g.nodes()  #
            for i in sorted_nodes:
                if i not in mapping.keys():
                    mapping[i] = len(mapping.keys())

        for i in range(len(graphs)):
            G = nx.MultiGraph()
            tmp_G = relabel_nodes(graphs[i], mapping=mapping)

            G.add_nodes_from(sorted(tmp_G.nodes()))
            G.add_edges_from(sorted(tmp_G.edges()))

            graphs[i] = G #relabel_nodes(graphs[i], mapping=mapping)

        print("Loaded {} graphs ".format(len(graphs)))
        adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs))
        return graphs, adj_matrices, mapping

    # mapping = {}
    # graphs = pickle.load(open(dataset_str + "/graphs.pkl", "rb"))
    # for g in graphs[:FLAGS.time_steps]:
    #     for i in sorted(g.nodes()):
    #         if i not in mapping.keys():
    #             mapping[i] = len(mapping.keys())
    # for i in range(len(graphs[:FLAGS.time_steps])):
    #     graphs[i] = relabel_nodes(graphs[i], mapping=mapping)
    #
    # print("Loaded {} graphs ".format(len(graphs[:FLAGS.time_steps])))
    # adj_matrices = list(map(lambda x: nx.adjacency_matrix(x), graphs[:FLAGS.time_steps]))
    # return graphs[:FLAGS.time_steps], adj_matrices, mapping


def load_feats(dataset_str):
    """ Load node attribute snapshots given the name of dataset (not used in experiments)"""
    if FLAGS.features == 'reid':
        features = pickle.load(open(dataset_str + "/numpy_reid_feats.pkl", "rb"))
    elif FLAGS.features == 'maskrcnn':
        features = pickle.load(open(dataset_str + "/numpy_maskrcnn_feats.pkl", "rb"))

    print("Loaded {} X matrices ".format(len(features)))
    return features


def sparse_to_tuple(sparse_mx):
    """Convert scipy sparse matrix to tuple representation (for tf feed dict)."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    def to_tuple_list(matrices):
        # Input is a list of matrices.
        coords = []
        values = []
        shape = [len(matrices)]
        for i in range(0, len(matrices)):
            mx = matrices[i]
            if not sp.isspmatrix_coo(mx):
                mx = mx.tocoo()
            # Create proper indices - coords is a numpy array of pairs of indices.
            coords_mx = np.vstack((mx.row, mx.col)).transpose()
            z = np.array([np.ones(coords_mx.shape[0]) * i]).T
            z = np.concatenate((z, coords_mx), axis=1)
            z = z.astype(int)
            coords.extend(z)
            values.extend(mx.data)

        shape.extend(matrices[0].shape)
        shape = np.array(shape).astype("int64")
        values = np.array(values).astype("float32")
        coords = np.array(coords)
        return coords, values, shape

    if isinstance(sparse_mx, list) and isinstance(sparse_mx[0], list):
        # Given a list of lists, convert it into a list of tuples.
        for i in range(0, len(sparse_mx)):
            sparse_mx[i] = to_tuple_list(sparse_mx[i])

    elif isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features, is_sparse=True):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)

    if is_sparse:
        return features.todense(), sparse_to_tuple(features)
    else:
        return features, features  # dense_to_tuple(features)


def normalize_graph_gcn(adj, to_tuple=True):
    """GCN-based normalization of adjacency matrix (scipy sparse format). Output is in tuple format"""
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(
        degree_mat_inv_sqrt).tocoo()
    if to_tuple:
        return sparse_to_tuple(adj_normalized)
    else:
        return adj_normalized.toarray()


def get_context_pairs_incremental(graph):
    return run_random_walks_n2v(graph, graph.nodes())


def get_context_pairs(graphs, num_time_steps):
    """ Load/generate context pairs for each snapshot through random walk sampling."""

    if FLAGS.window > 0:
        # For self-attention over a window of ``FLAGS.window'' time steps.
        #min_t = max(num_time_steps - FLAGS.window - 1, 0)
        min_t = 0

    load_path = "{}/{}/train_pairs_n2v_{}.pkl".format(
        FLAGS.dataset_cache_path, FLAGS.dataset, str(num_time_steps - 2))
    try:
        context_pairs_train = dill.load(open(load_path, 'rb'))
        print("Loaded context pairs from pkl file directly")
    except (IOError, EOFError):
        print("Computing training pairs ...")
        context_pairs_train = []
        for i in range(min_t, num_time_steps):
            if i > 0:
                if len(graphs[i - 1].nodes()) == len(graphs[i].nodes()) and len(graphs[i - 1].edges()) == len(graphs[i].edges()):
                    context_pairs_train.append(context_pairs_train[i - 1])
                else:
                    context_pairs_train.append(
                        run_random_walks_n2v(graphs[i], graphs[i].nodes()))
            else:
                context_pairs_train.append(
                    run_random_walks_n2v(graphs[i], graphs[i].nodes()))

            print("Getting context pairs time_step ", i, end="\r")
        dill.dump(context_pairs_train, open(load_path, 'wb'))
        print("Saved pairs")

    return context_pairs_train


def get_evaluation_data(adjs, num_time_steps, dataset, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """ Load train/val/test examples to evaluate link prediction performance"""

    train_edges_all = []
    train_edges_false_all = []
    val_edges_all = []
    val_edges_false_all = []
    test_edges_all = []
    test_edges_false_all = []

    for i in range(FLAGS.window, num_time_steps, FLAGS.window):
        eval_idx = i - 2
        eval_path = "{}/{}/eval-{}/eval_{}.npz".format(FLAGS.dataset_cache_path, FLAGS.dataset, FLAGS.window, str(eval_idx))
        try:
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                np.load(eval_path, encoding='bytes', allow_pickle=True)['data']
            print("Loaded eval data", i,  end="\r")
        except IOError:
            next_adjs = adjs[eval_idx + 1]
            print("Generating and saving eval data ....")
            train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = \
                create_data_splits(adjs[eval_idx], next_adjs,
                                   val_mask_fraction=val_mask_fraction, test_mask_fraction=test_mask_fraction)
            if not os.path.exists(os.path.dirname(eval_path)):
                os.makedirs(os.path.dirname(eval_path))

            np.savez(eval_path, data=np.array([train_edges, train_edges_false, val_edges, val_edges_false,
                                               test_edges, test_edges_false]))

        train_edges_all.append(train_edges)
        train_edges_false_all.append(train_edges_false)
        val_edges_all.append(val_edges)
        val_edges_false_all.append(val_edges_false)
        test_edges_all.append(test_edges)
        test_edges_false_all.append(test_edges_false)

    return train_edges_all, train_edges_false_all, val_edges_all, val_edges_false_all, test_edges_all, test_edges_false_all


def create_data_splits(adj, next_adj, val_mask_fraction=0.2, test_mask_fraction=0.6):
    """In: (adj, next_adj) along with test and val fractions. For link prediction (on all links), all links in
    next_adj are considered positive examples.
    Out: list of positive and negative pairs for link prediction (train/val/test)"""
    edges_all = sparse_to_tuple(next_adj)[0]  # All edges in original adj.
    # Remove diagonal elements
    adj = adj - \
        sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    assert np.diag(adj.todense()).sum() == 0
    if next_adj is None:
        raise ValueError('Next adjacency matrix is None')

    edges_next = np.array(
        list(set(nx.from_scipy_sparse_matrix(next_adj).edges())))
    edges = []   # Constraint to restrict new links to existing nodes. Assume new nodes at higher indices
    for e in edges_next:
        if e[0] < adj.shape[0] and e[1] < adj.shape[0]:
            edges.append(e)
    edges = np.array(edges)

    def ismember(a, b, tol=5):
        rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
        return np.any(rows_close)

    all_edge_idx = list(range(edges.shape[0]))
    np.random.shuffle(all_edge_idx)
    num_test = int(np.floor(edges.shape[0] * test_mask_fraction))
    num_val = int(np.floor(edges.shape[0] * val_mask_fraction))
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    test_edges = edges[test_edge_idx]
    val_edge_idx = all_edge_idx[:num_val]
    val_edges = edges[val_edge_idx]

    train_edges = np.delete(edges, np.hstack(
        [test_edge_idx, val_edge_idx]), axis=0)

    edges_new = []

    for e in edges_next:
        if not ismember(e, edges):
            edges_new.append(e)

    if len(test_edges) > 0:
        if len(edges_new) > 0:
            test_edges = np.concatenate( (np.asarray(edges_new, dtype=np.int64), test_edges), axis=0)
    else:
        if len(edges_new) > 0:
            test_edges = np.asarray(edges_new, dtype=np.int64)

        # Create train edges.
    train_edges_false = []
    while len(train_edges_false) < len(train_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if train_edges_false:
            if ismember([idx_j, idx_i], np.array(train_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(train_edges_false)):
                continue
        train_edges_false.append([idx_i, idx_j])

    # Create test edges.
    test_edges_false = []
    while len(test_edges_false) < len(test_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue
        if test_edges_false:
            if ismember([idx_j, idx_i], np.array(test_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(test_edges_false)):
                continue
        test_edges_false.append([idx_i, idx_j])

    # Create val edges.
    val_edges_false = []
    while len(val_edges_false) < len(val_edges):
        idx_i = np.random.randint(0, adj.shape[0])
        idx_j = np.random.randint(0, adj.shape[0])
        if idx_i == idx_j:
            continue
        if ismember([idx_i, idx_j], edges_all):
            continue
        if ismember([idx_j, idx_i], edges_all):
            continue

        if val_edges_false:
            if ismember([idx_j, idx_i], np.array(val_edges_false)):
                continue
            if ismember([idx_i, idx_j], np.array(val_edges_false)):
                continue
        val_edges_false.append([idx_i, idx_j])
    try:
        assert ~ismember(test_edges_false, edges_all)
        assert ~ismember(val_edges_false, edges_all)
        assert ~ismember(val_edges, train_edges)
        assert ~ismember(test_edges, train_edges)
        assert ~ismember(val_edges, test_edges)
    except:
        pass
    print("# train examples: ", len(train_edges), len(train_edges_false))
    print("# val examples:", len(val_edges), len(val_edges_false))
    print("# test examples:", len(test_edges), len(test_edges_false))

    return list(train_edges), train_edges_false, list(val_edges), val_edges_false, list(test_edges), test_edges_false

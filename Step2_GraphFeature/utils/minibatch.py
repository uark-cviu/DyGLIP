from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import random

flags = tf.app.flags
FLAGS = flags.FLAGS


class NodeMinibatchIterator(object):
    """
    This minibatch iterator iterates over nodes to sample context pairs for a batch of nodes.

    graphs -- list of networkx graphs
    features -- list of (scipy) sparse node attribute matrices
    adjs -- list of adj matrices (of the graphs)
    placeholders -- standard tensorflow placeholders object for feeding
    num_time_steps -- number of graphs to train +1
    context_pairs -- list of (target, context) pairs obtained from random walk sampling.
    batch_size -- size of the minibatches (# nodes)
    """

    def __init__(self, graphs, features, adjs, placeholders, num_time_steps, context_pairs=None, batch_size=1, node_per_batch=512):

        self.graphs = graphs
        self.features = features
        self.adjs = adjs
        self.placeholders = placeholders
        self.nodes_per_batch = node_per_batch
        self.batch_size = batch_size
        self.batch_num = 0
        self.num_time_steps = num_time_steps
        self.degs = self.construct_degs()
        self.context_pairs = context_pairs
        self.max_positive = FLAGS.neg_sample_size
        # all nodes in the graph.

        self.train_nodes = []
        self.train_graph_idx = []
        for i in range(0, self.num_time_steps - FLAGS.window + 1, FLAGS.window):
            self.train_nodes.append(self.graphs[i + FLAGS.window - 1].nodes())
            self.train_graph_idx.append(i)
        print("# train nodes", len(self.train_nodes))

    def construct_degs(self):
        """ Compute node degrees in each graph snapshot."""
        degs = []
        for i in range(0, self.num_time_steps):
            G = self.graphs[i]
            deg = np.zeros((len(G.nodes()),))
            for nodeid in G.nodes():
                neighbors = np.array(list(G.neighbors(nodeid)))
                deg[nodeid] = len(neighbors)
            degs.append(deg)
        # min_t = 0
        # if FLAGS.window > 0:
        #     min_t = max(self.num_time_steps - FLAGS.window - 1, 0)
        return degs #[min_t:]

    def end(self):
        return self.batch_num * self.batch_size >= len(self.train_nodes)

    def batch_feed_dict(self, batch_nodes, batch_idx):
        """ Feed dict with (a) node pairs, (b) list of attribute matrices (c) list of snapshot adjs and metadata"""
        node_1_all = []
        node_2_all = []
        node_3_all = []
        min_t = 0
        if FLAGS.window > 0:
            # For self-attention over a window of ``FLAGS.window'' time steps.

            if self.batch_size == 1:
                min_t = batch_idx #max(self.num_time_steps - FLAGS.window - 1, 0)
                max_t = batch_idx + FLAGS.window
            else:
                min_t = 0 #max(self.num_time_steps - FLAGS.window - 1, 0)
                max_t = FLAGS.window

        if self.context_pairs is not None:

            if isinstance(self.context_pairs[0], tuple):
                for t in range(min_t, max_t):
                    node_1 = []
                    node_2 = []
                    node_3 = []

                    for n in batch_nodes:

                        if len(self.context_pairs[t][0][n]) > self.max_positive:
                            node_1.extend([n] * self.max_positive)
                            node_2.extend(np.random.choice(
                                self.context_pairs[t][0][n], self.max_positive, replace=False))
                            node_3.extend(np.random.choice(
                                self.context_pairs[t][1][n], self.max_positive, replace=False))
                        else:
                            node_1.extend([n] * len(self.context_pairs[t][0][n]))
                            node_2.extend(self.context_pairs[t][0][n])
                            node_3.extend(self.context_pairs[t][1][n])

                    assert len(node_1) == len(node_2)
                    assert len(node_1) <= self.nodes_per_batch * self.max_positive

                    node_1_all.append(node_1)
                    node_2_all.append(node_2)
                    node_3_all.append(node_3)
            else:
                for t in range(min_t, max_t):
                    node_1 = []
                    node_2 = []

                    for n in batch_nodes:

                        if len(self.context_pairs[t][n]) > self.max_positive:
                            node_1.extend([n] * self.max_positive)
                            node_2.extend(np.random.choice(
                                self.context_pairs[t][n], self.max_positive, replace=False))
                        else:
                            node_1.extend([n] * len(self.context_pairs[t][n]))
                            node_2.extend(self.context_pairs[t][n])

                    assert len(node_1) == len(node_2)
                    assert len(node_1) <= self.nodes_per_batch * self.max_positive

                    node_1_all.append(node_1)
                    node_2_all.append(node_2)

        feed_dict = dict()

        if len(node_1_all) > 0:
            feed_dict.update({self.placeholders['node_1'][t - min_t]: node_1_all[t - min_t]
                          for t in range(min_t, max_t)})

        if len(node_2_all) > 0:
            feed_dict.update({self.placeholders['node_2'][t - min_t]: node_2_all[t - min_t]
                          for t in range(min_t, max_t)})

        if len(node_3_all) > 0:
            feed_dict.update({self.placeholders['node_3'][t - min_t]: node_3_all[t - min_t]
                          for t in range(min_t, max_t)})

        feed_dict.update({self.placeholders['features'][t - min_t]: self.features[t] for t in range(min_t, max_t)})
        feed_dict.update({self.placeholders['adjs'][t - min_t]: self.adjs[t]
                          for t in range(min_t, max_t)})

        feed_dict.update(
            {self.placeholders['batch_nodes']: np.array(batch_nodes).astype(np.int32)})

        # min_nodes = len(batch_nodes[0])
        #
        # for n in batch_nodes:
        #     if(len(n) < min_nodes):
        #         min_nodes = len(n)

        # for t in range(min_t, max_t):
        #     batch_node_1 = []
        #     batch_node_2 = []
        #
        #     for b, nodes in enumerate(batch_nodes):
        #         time_step = batch_idx[b] + t
        #
        #         node_1 = []
        #         node_2 = []
        #
        #         for n in nodes:
        #             if len(self.context_pairs[time_step][n]) > self.max_positive:
        #                 node_1.extend([n] * self.max_positive)
        #                 node_2.extend(np.random.choice(
        #                     self.context_pairs[time_step][n], self.max_positive, replace=False))
        #             else:
        #                 node_1.extend([n] * len(self.context_pairs[time_step][n]))
        #                 node_2.extend(self.context_pairs[time_step][n])
        #
        #         assert len(node_1) == len(node_2)
        #         assert len(node_1) <= self.nodes_per_batch * self.max_positive
        #
        #         batch_node_1.append(node_1)
        #         batch_node_2.append(node_2)
        #
        #     feed_dict = dict()
        #     node_1_all.append(batch_node_1)
        #     node_2_all.append(batch_node_2)

        # dict = {}
        # for t in range(min_t, max_t):
        #     batch_features = []
        #     batch_adjs = []
        #     for b in range(self.batch_size):
        #         time_step = batch_idx[b] + t
        #
        #         feed_dict.update({self.placeholders['node_1'][b][t - min_t]: node_1_all[t - min_t][b]})
        #         feed_dict.update({self.placeholders['node_2'][b][t - min_t]: node_2_all[t - min_t][b]})
        #
        #         batch_features.append(self.features[time_step])
        #         batch_adjs.append(self.adjs[time_step])
        #
        #     feed_dict.update({self.placeholders['features'][t - min_t]: self.features[time_step]})
        #     feed_dict.update({self.placeholders['adjs'][t - min_t]: self.adjs[time_step]})

        return feed_dict

    def num_training_batches(self):
        """ Compute the number of training batches (using batch size)"""
        return len(self.train_nodes) // self.batch_size + 1

    def next_minibatch_feed_dict(self):
        """ Return the feed_dict for the next minibatch (in the current epoch) with random shuffling"""
        start_idx = self.batch_num * self.batch_size
        self.batch_num += 1
        end_idx = min(start_idx + self.batch_size, len(self.train_nodes))
        batch_nodes = self.train_nodes[start_idx: end_idx]
        batch_idx = self.train_graph_idx[start_idx: end_idx]

        if self.batch_size == 1:
            return self.batch_feed_dict(batch_nodes[0], batch_idx[0])
        else:
            return self.batch_feed_dict(batch_nodes, batch_idx)

    def shuffle(self):
        """ Re-shuffle the training set.
            Also reset the batch number.
        """
        for i in range(len(self.train_nodes)):
            self.train_nodes[i] = np.random.permutation(self.train_nodes[i])

        mapIndexPosition = list(zip(self.train_nodes, self.train_graph_idx))

        random.shuffle(mapIndexPosition)

        self.train_nodes, self.train_graph_idx = zip(*mapIndexPosition)

        self.train_nodes = list(self.train_nodes)
        self.train_graph_idx = list(self.train_graph_idx)

        self.batch_num = 0

    def test_reset(self, full=True):
        """ Reset batch number"""
        #self.train_nodes = self.graphs[self.num_time_steps-1].nodes()
        self.train_nodes = []
        self.train_graph_idx = []
        for i in range(FLAGS.window, self.num_time_steps, FLAGS.window):
            self.train_nodes.append(self.graphs[i].nodes())
            self.train_graph_idx.append(i - FLAGS.window)
        if full:
            if self.train_graph_idx[-1] != self.num_time_steps - FLAGS.window:
                self.train_nodes.append(self.graphs[-1].nodes())
                self.train_graph_idx.append(self.num_time_steps - FLAGS.window)
        self.batch_num = 0

import pickle
import numpy as np
import networkx as nx
import os.path as osp
from itertools import combinations
import glob

path = "/data/Processed/"
datasets = glob.glob(osp.join(path, "*"))
for data in datasets:
    # if (data.split("/")[-1] == "EPLF" or data.split("/")[-1] == "PET" or data.split("/")[-1] == "CAMPUS"): continue
    scenes = glob.glob(osp.join(data, "*"))
    for scene in scenes:
        print(scene)
        with open(osp.join(scene, "reid_feats.pkl"), "rb") as f:
            feats = pickle.load(f)

        frames = set()
        graphs = []
        nodes = []
        edges = {}
        hits = {}

        for video in sorted(feats.keys()):
            frames.update(feats[video].keys())
            # print(len(frames), min(frames))

        frames = sorted(frames)

        for frid in frames:
            graph = nx.MultiGraph()
            for video in sorted(feats.keys()):
                if frid in feats[video].keys():
                    for trid in sorted(feats[video][frid].keys()):
                        if "_".join((video, trid)) not in hits.keys():
                            hits["_".join((video, trid))] = {0: frid}
                            times = sorted(list(hits["_".join((video, trid))].keys()))[-1]
                        else:
                            times = sorted(list(hits["_".join((video, trid))].keys()))[-1]
                            _count = hits["_".join((video, trid))][times]
                            if frid - _count > 3:
                                hits["_".join((video, trid))] = {times + 1: frid}
                                times = sorted(list(hits["_".join((video, trid))].keys()))[-1]
                            else:
                                hits["_".join((video, trid))] = {times: frid}
                        node_name = "_".join((video, trid, str(times)))
                        if node_name not in nodes:
                            nodes.append(node_name)
                        if trid not in edges.keys():
                            edges[trid] = []
                        if node_name not in edges[trid]:
                            edges[trid].append(node_name)

            for n in nodes:
                graph.add_node(n)
            for trid in edges.keys():
                graph.add_edges_from([(i, j) for i, j in combinations(edges[trid], 2) if not graph.has_edge(i, j)])
            graphs.append(graph)
            prev_graph = graph
            print(frid, end="\r")

        with open(osp.join(data, scene, "graphs.pkl"), "wb") as f:
            pickle.dump(graphs, f)

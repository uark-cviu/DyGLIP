from __future__ import division, print_function
from sklearn.metrics import roc_auc_score
import numpy as np
from sklearn import linear_model
from collections import defaultdict
import random
import pickle
from tqdm import tqdm

np.random.seed(123)
operatorTypes = ["HAD"]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def write_to_csv(test_results, model_name, dataset, time_steps, mod='val'):
    """Output result scores to a csv file for result logging"""
    # with open(output_name, 'a+') as f:
    for op in test_results:
        print("{} results ({})".format(model_name, mod), test_results[op])
        _, best_auc = test_results[op]
            # f.write("{},{},{},{},{},{},{}\n".format(
            #     dataset, time_steps, model_name, op, mod, "AUC", best_auc))


def get_link_score(fu, fv, operator):
    """Given a pair of embeddings, compute link feature based on operator (such as Hadammad product, etc.)"""
    fu = np.array(fu)
    fv = np.array(fv)
    if operator == "HAD":
        return np.multiply(fu, fv)
    else:
        raise NotImplementedError


def get_link_feats(links, source_embeddings, target_embeddings, operator):
    """Compute link features for a list of pairs"""
    features = []
    for l in links:
        a, b = l[0], l[1]
        f = get_link_score(
            source_embeddings[a], target_embeddings[b], operator)
        features.append(f)
    return features


def get_random_split(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg):
    """ Randomly split a given set of train, val and test examples"""
    all_data_pos = []
    all_data_neg = []

    all_data_pos.extend(train_pos)
    all_data_neg.extend(train_neg)
    all_data_pos.extend(test_pos)
    all_data_neg.extend(test_neg)

    # re-define train_pos, train_neg, test_pos, test_neg.
    random.shuffle(all_data_pos)
    random.shuffle(all_data_neg)

    train_pos = all_data_pos[:int(0.2 * len(all_data_pos))]
    train_neg = all_data_neg[:int(0.2 * len(all_data_neg))]

    test_pos = all_data_pos[int(0.2 * len(all_data_pos)):]
    test_neg = all_data_neg[int(0.2 * len(all_data_neg)):]
    print("# train :", len(train_pos) + len(train_neg), "# val :", len(val_pos) + len(val_neg),
          "#test :", len(test_pos) + len(test_neg))
    return train_pos, train_neg, val_pos, val_neg, test_pos, test_neg


def evaluate_classifier_list(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds, model_path, pretrained=False, epoch=0, test=False):
    """Downstream logistic regression classifier to evaluate link prediction"""
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])

    n_list = len(source_embeds)
    test_auc = 0
    val_auc = 0

    n_list_test_valid = n_list
    n_list_val_valid = n_list

    for i in range(n_list):

        test_score = get_roc_score_t(
            test_pos[i], test_neg[i], source_embeds[i], target_embeds[i])

        if test_score >= 0:
            test_auc += test_score
        else:
            n_list_test_valid -= 1
        if not test:
            val_score = get_roc_score_t(val_pos[i], val_neg[i], source_embeds[i], target_embeds[i])

            if val_score >= 0:
                val_auc += val_score
            else:
                n_list_val_valid -= 1

    # Compute AUC based on sigmoid(u^T v) without classifier training.
    test_results['SIGMOID'].extend([test_auc / n_list_test_valid, test_auc / n_list_test_valid])
    if not test:
        val_results['SIGMOID'].extend([val_auc / n_list_val_valid, val_auc / n_list_val_valid])

    test_pred_true = defaultdict(lambda: [])
    val_pred_true = defaultdict(lambda: [])

    for operator in operatorTypes:
        train_data = None
        train_labels = None

        val_data = None
        val_labels = None

        test_data = None
        test_labels = None

        #for i in range(n_list):
        for i in tqdm(range(n_list)):
            #print(i)

            if not test and not pretrained:
                train_pos_feats = np.array(get_link_feats(
                    train_pos[i], source_embeds[i], target_embeds[i], operator))
                train_neg_feats = np.array(get_link_feats(
                    train_neg[i], source_embeds[i], target_embeds[i], operator))
            if not test:
                val_pos_feats = np.array(get_link_feats(
                    val_pos[i], source_embeds[i], target_embeds[i], operator))
                val_neg_feats = np.array(get_link_feats(
                    val_neg[i], source_embeds[i], target_embeds[i], operator))
            test_pos_feats = np.array(get_link_feats(
                test_pos[i], source_embeds[i], target_embeds[i], operator))
            test_neg_feats = np.array(get_link_feats(
                test_neg[i], source_embeds[i], target_embeds[i], operator))

            if not test and not pretrained:
                train_pos_labels = np.array([1] * len(train_pos_feats))
                train_neg_labels = np.array([-1] * len(train_neg_feats))

            if not test:
                val_pos_labels = np.array([1] * len(val_pos_feats))
                val_neg_labels = np.array([-1] * len(val_neg_feats))

            test_pos_labels = np.array([1] * len(test_pos_feats))
            test_neg_labels = np.array([-1] * len(test_neg_feats))

            if not test and not pretrained:
                if train_data is None:
                    train_data = np.vstack((train_pos_feats, train_neg_feats))
                else:
                    train_data = np.vstack((train_data, train_pos_feats, train_neg_feats))

                if train_labels is None:
                    train_labels = np.append(train_pos_labels, train_neg_labels)
                else:
                    train_labels = np.append(train_labels, np.append(train_pos_labels, train_neg_labels))

            if not test:
                if val_data is None or val_data.size == 0:
                    if val_pos_feats.shape[0] > 0 and val_neg_feats.shape[0] > 0:
                        val_data = np.vstack((val_pos_feats, val_neg_feats))
                elif val_data.size > 0:
                    if val_pos_feats.shape[0] > 0 and val_neg_feats.shape[0] > 0:
                        val_data = np.vstack((val_data, val_pos_feats, val_neg_feats))

                if val_labels is None or val_labels.size == 0:
                    val_labels = np.append(val_pos_labels, val_neg_labels)
                elif val_labels.size > 0:
                    val_labels = np.append(val_labels, np.append(val_pos_labels, val_neg_labels))

            if test_data is None or test_data.size == 0:
                if test_pos_feats.shape[0] > 0 and test_neg_feats.shape[0] > 0:
                    test_data = np.vstack((test_pos_feats, test_neg_feats))
            elif test_data.size > 0:
                if test_pos_feats.shape[0] > 0 and test_neg_feats.shape[0] > 0:
                    test_data = np.vstack((test_data, test_pos_feats, test_neg_feats))

            if test_labels is None or test_labels.size == 0:
                test_labels = np.append(test_pos_labels, test_neg_labels)
            elif test_labels.size > 0:
                test_labels = np.append(test_labels, np.append(test_pos_labels, test_neg_labels))

        if not pretrained:
            logistic = linear_model.LogisticRegression()
            logistic.fit(train_data, train_labels)
            pickle.dump(logistic, open("{}/logistic-{}.pkl".format(model_path, epoch), "wb"))
        else:
            logistic = pickle.load(open("{}/logistic-{}.pkl".format(model_path, epoch), "rb"))

        test_predict = logistic.predict_proba(test_data)[:, 1]
        if not test:
            val_predict = logistic.predict_proba(val_data)[:, 1]

        test_roc_score = roc_auc_score(test_labels, test_predict)
        if not test:
            val_roc_score = roc_auc_score(val_labels, val_predict)
        if not test:
            val_results[operator].extend([val_roc_score, val_roc_score])
        test_results[operator].extend([test_roc_score, test_roc_score])
        if not test:
            val_pred_true[operator].extend(zip(val_predict, val_labels))
        test_pred_true[operator].extend(zip(test_predict, test_labels))
    if not test:
        return val_results, test_results, val_pred_true, test_pred_true
    else:
        return [], test_results, [], test_pred_true


def predict_link(pairs, source_embeds, target_embeds, model_path=None, epoch=0):

    n_list = len(source_embeds)

    for operator in operatorTypes:
        test_data = None

        if model_path is not None:
            for i in range(n_list):

                test_feats = np.array(get_link_feats(
                    pairs[i], source_embeds[i], target_embeds[i], operator))

                if test_data is None:
                    test_data = test_feats
                else:
                    test_data = np.vstack((test_data, test_feats))

            logistic = pickle.load(open("{}/logistic-{}.pkl".format(model_path, epoch), "rb"))

            test_predict = logistic.predict_proba(test_data)[:, 1]
        else:
            test_predict = []
            for i in range(n_list):

                adj_rec = np.dot(source_embeds[i], target_embeds[i].T)

                pred = []
                for e in pairs:
                    pred.append(sigmoid(adj_rec[e[0], e[1]]))

            test_predict.append(pred)

    return test_predict


def evaluate_classifier(train_pos, train_neg, val_pos, val_neg, test_pos, test_neg, source_embeds, target_embeds, model_path, pretrained=False, epoch=0, test=False):
    """Downstream logistic regression classifier to evaluate link prediction"""
    test_results = defaultdict(lambda: [])
    val_results = defaultdict(lambda: [])

    test_auc = get_roc_score_t(
        test_pos, test_neg, source_embeds, target_embeds)
    if not test:
        val_auc = get_roc_score_t(val_pos, val_neg, source_embeds, target_embeds)

    # Compute AUC based on sigmoid(u^T v) without classifier training.
    test_results['SIGMOID'].extend([test_auc, test_auc])
    if not test:
        val_results['SIGMOID'].extend([val_auc, val_auc])

    test_pred_true = defaultdict(lambda: [])
    val_pred_true = defaultdict(lambda: [])

    for operator in operatorTypes:
        train_pos_feats = np.array(get_link_feats(
            train_pos, source_embeds, target_embeds, operator))
        train_neg_feats = np.array(get_link_feats(
            train_neg, source_embeds, target_embeds, operator))
        if not test:
            val_pos_feats = np.array(get_link_feats(
                val_pos, source_embeds, target_embeds, operator))
            val_neg_feats = np.array(get_link_feats(
                val_neg, source_embeds, target_embeds, operator))
        test_pos_feats = np.array(get_link_feats(
            test_pos, source_embeds, target_embeds, operator))
        test_neg_feats = np.array(get_link_feats(
            test_neg, source_embeds, target_embeds, operator))

        train_pos_labels = np.array([1] * len(train_pos_feats))
        train_neg_labels = np.array([-1] * len(train_neg_feats))
        if not test:
            val_pos_labels = np.array([1] * len(val_pos_feats))
            val_neg_labels = np.array([-1] * len(val_neg_feats))

        test_pos_labels = np.array([1] * len(test_pos_feats))
        test_neg_labels = np.array([-1] * len(test_neg_feats))
        train_data = np.vstack((train_pos_feats, train_neg_feats))
        train_labels = np.append(train_pos_labels, train_neg_labels)
        if not test:
            val_data = np.vstack((val_pos_feats, val_neg_feats))
            val_labels = np.append(val_pos_labels, val_neg_labels)

        test_data = np.vstack((test_pos_feats, test_neg_feats))
        test_labels = np.append(test_pos_labels, test_neg_labels)

        if not pretrained:
            logistic = linear_model.LogisticRegression()
            logistic.fit(train_data, train_labels)
            pickle.dump(logistic, open("{}/logistic-{}.pkl".format(model_path, epoch), "wb"))
        else:
            logistic = pickle.load(open("{}/logistic-{}.pkl".format(model_path, epoch), "rb"))

        test_predict = logistic.predict_proba(test_data)[:, 1]
        if not test:
            val_predict = logistic.predict_proba(val_data)[:, 1]

        test_roc_score = roc_auc_score(test_labels, test_predict)
        if not test:
            val_roc_score = roc_auc_score(val_labels, val_predict)
        if not test:
            val_results[operator].extend([val_roc_score, val_roc_score])
        test_results[operator].extend([test_roc_score, test_roc_score])
        if not test:
            val_pred_true[operator].extend(zip(val_predict, val_labels))
        test_pred_true[operator].extend(zip(test_predict, test_labels))
    if not test:
        return val_results, test_results, val_pred_true, test_pred_true
    else:
        return [], test_results, [], test_pred_true


def get_roc_score_t(edges_pos, edges_neg, source_emb, target_emb):
    """Given test examples, edges_pos: +ve edges, edges_neg: -ve edges, return ROC scores for a given snapshot"""

    # Predict on test set of edges
    adj_rec = np.dot(source_emb, target_emb.T)
    pred = []
    pos = []
    for e in edges_pos:
        pred.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(1.0)

    pred_neg = []
    neg = []
    for e in edges_neg:
        pred_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(0.0)

    pred_all = np.hstack([pred, pred_neg])
    labels_all = np.hstack([np.ones(len(pred)), np.zeros(len(pred_neg))])

    if len(labels_all) > 0 and len(pred_all) > 0:
        roc_score = roc_auc_score(labels_all, pred_all)
    else:
        roc_score = -1
    return roc_score

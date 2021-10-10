#!/usr/bin/env python

import os
import sys
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import faiss
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


def read_train_test_data(dataset, k, directed=True, inputfold="./data/train_test"):
    train_X = []
    train_y = []
    filename = lambda dataset, k, mode: '{3}/{0}/{0}_maxC_{2}{1}.edgelist'.format(dataset, k, mode, inputfold)
    with open(filename(dataset, k, "train")) as f:
        for line in f:
            if "#" in line: continue
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1) / 2)
            train_X.append((i, j))
            train_y.append(flag)
            if not directed:
                train_X.append((j, i))
                train_y.append(flag)
    test_X = []
    test_y = []
    with open(filename(dataset, k, "test")) as f:
        for line in f:
            if "#" in line: continue
            i, j, flag = line.split()
            i = int(i)
            j = int(j)
            flag = int(flag)
            flag = int((flag + 1) / 2)
            test_X.append((i, j))
            test_y.append(flag)
    return np.array(train_X), np.array(train_y), np.array(test_X), np.array(test_y)


def load_features(dataset, k, mode="", logfold="./loggings", inputfold="./data/train_test", directed=True):
    dirname = "{0}/{1}/".format(logfold, dataset)
    fpath = os.path.join(dirname, '{0}_{1}_embedding{2}.txt'.format(dataset, k, mode))
    embeddings = np.loadtxt(fpath)

    train_X, train_y, test_X, test_y = read_train_test_data(dataset, k, directed, inputfold)
    train_X1 = []
    test_X1 = []
    for i, j in train_X:
        train_X1.append(np.concatenate([embeddings[i], embeddings[j]]))
    for i, j in test_X:
        test_X1.append(np.concatenate([embeddings[i], embeddings[j]]))
    train_X1, test_X1 = np.asarray(train_X1), np.asarray(test_X1)
    return train_X1, train_y, test_X1, test_y


def get_weights(dist):
    dist_copy = np.empty_like(dist)
    for point_dist_i, point_dist in enumerate(dist):
        if 0. in point_dist:
            dist_copy[point_dist_i] = point_dist == 0.
        else:
            dist_copy[point_dist_i] = 1. / point_dist
    return dist_copy


def get_probability(X, _y, weights, neigh_ind, sample_weights=None):
    _y = _y.reshape((-1, 1))
    classes_ = [np.array([0, 1])]
    n_queries = len(X)
    all_rows = np.arange(X.shape[0])
    probabilities = []
    if sample_weights is not None:
        sample_weights = sample_weights[neigh_ind]
    for k, classes_k in enumerate(classes_):
        pred_labels = _y[:, k][neigh_ind].astype(np.int)
        proba_k = np.zeros((n_queries, classes_k.size))
        for i, idx in enumerate(pred_labels.T):
            if sample_weights is not None:
                proba_k[all_rows, idx] += weights[:, i] * sample_weights[:, i]
            else:
                proba_k[all_rows, idx] += weights[:, i]
        normalizer = proba_k.sum(axis=1)[:, np.newaxis]
        normalizer[normalizer == 0.0] = 1.0
        proba_k /= normalizer
        probabilities.append(proba_k)
    return probabilities[0]


def classifiers(train_X, train_y, test_X, test_y, K=8, probas=None):
    if probas is None:
        probas = [0.5]
    logistic_function = linear_model.LogisticRegression()
    logistic_function.fit(train_X, train_y)
    proba_lr = logistic_function.predict_proba(test_X)

    train_X = train_X.astype(np.float32)
    test_X = test_X.astype(np.float32)
    dims = train_X.shape[1]
    index = faiss.IndexFlatL2(dims)
    res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(res, 0, index)
    index.train(train_X)
    index.add(train_X)
    neigh_distance, neigh_indices = index.search(test_X, K)
    weights = get_weights(neigh_distance)
    proba_knn = get_probability(test_X, train_y, weights, neigh_indices)

    for i in probas:
        print("proba", i, "=" * 20, i, "=" * 20)
        interpolation_evaluation(test_y, proba_lr, proba_knn, i)
        sys.stdout.flush()


def interpolation_evaluation(test_y, proba_lr, proba_knn, alpha):
    pred_p = proba_knn * alpha + proba_lr * (1 - alpha)
    pred = np.argmax(pred_p, axis=1)

    f1_score_ma = metrics.f1_score(test_y, pred, average='macro')
    f1_score_mi = metrics.f1_score(test_y, pred, average='micro')
    f1_score_wt = metrics.f1_score(test_y, pred, average='weighted')
    f1_score_bi = metrics.f1_score(test_y, pred, average='binary')
    auc = metrics.roc_auc_score(test_y, pred_p[:, 1])

    print(metrics.confusion_matrix(test_y, pred))
    print('f1_mi', f1_score_mi,
          'f1_ma', f1_score_ma,
          'f1_wt', f1_score_wt,
          'f1_bi', f1_score_bi,
          'auc', auc,
          )


if __name__ == "__main__":
    model = "SPR"

    dataset = "bitcoinAlpha"
    # dataset = "bitcoinOTC"
    # dataset = "slashdot"
    # dataset = "epinions"

    inputfold, logfold = "./data/train_test", "./loggings"
    mode = ""

    print(logfold, dataset)
    probas = [0.5, 0.0]  # 0.0 denotes prediction results by logistic regression classifier.
    for k in range(5):
        train_X1, train_y, test_X1, test_y = load_features(dataset, k, mode=mode, logfold=logfold, inputfold=inputfold)
        for Knn in [8]:
            print("{}_{} {} K:{}".format(dataset, k, model, Knn))
            classifiers(train_X1, train_y, test_X1, test_y, K=Knn, probas=probas)
            print("#" * 50)
    print("=" * 50)

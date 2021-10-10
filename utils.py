import argparse
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import pickle


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_undirected_networks(file_name, undirected=True):
    links = {}
    with open(file_name) as fp:
        n, m = [int(val) for val in fp.readline().split()[-2:]]
        for line in fp:
            line = line.strip()
            if line == "" or "#" in line:
                continue
            rater, rated, sign = [int(val) for val in line.split()]
            assert (sign != 0)
            sign = 1 if sign > 0 else -1

            if not undirected:
                edge1 = (rater, rated)
                if edge1 not in links:
                    links[edge1] = sign
                elif links[edge1] == sign:
                    pass
                else:
                    links[edge1] = -1
                continue

            edge1, edge2 = (rater, rated), (rated, rater)
            if edge1 not in links:
                links[edge1], links[edge2] = sign, sign
            elif links[edge1] == sign:
                pass
            else:
                links[edge1], links[edge2] = -1, -1

    adj_lists_pos, adj_lists_neg = defaultdict(set), defaultdict(set)
    num_edges_pos, num_edges_neg = 0, 0
    for (i, j), s in links.items():
        if s > 0:
            adj_lists_pos[i].add(j)
            num_edges_pos += 1
        else:
            adj_lists_neg[i].add(j)
            num_edges_neg += 1
    num_edges_pos /= 2
    num_edges_neg /= 2

    return n, [num_edges_pos, num_edges_neg], adj_lists_pos, adj_lists_neg


def read_in_feature_data(feature_train, features_dims):
    feat_data = pickle.load(open(feature_train, "rb"))
    if features_dims is not None:
        feat_data = feat_data[:, :features_dims]
    num_nodes, num_feats = feat_data.shape
    feat_data = StandardScaler().fit_transform(feat_data)
    return num_feats, feat_data


def load_data(args):
    net_train, feature_train, net_test, features_dims = \
        args['net_train'], args['features_train'], args['net_test'], args['feature_dim']

    num_nodes, num_edges, train_pos, train_neg = load_undirected_networks(net_train, undirected=True)
    _, _, train_pos_dir, train_neg_dir = load_undirected_networks(net_train, undirected=False)
    _, _, test_pos_dir, test_neg_dir = load_undirected_networks(net_test, undirected=False)
    num_feats, feat_data = read_in_feature_data(feature_train, features_dims)
    return num_nodes, num_edges, train_pos, train_neg, train_pos_dir, train_neg_dir, test_pos_dir, test_neg_dir, num_feats, feat_data


def calculate_class_weights(num_pos, num_neg):
    num_E = num_pos + num_neg
    w_pos_neg = 1
    w_pos = round(w_pos_neg * num_neg / num_E, 2)
    w_neg = round(w_pos_neg - w_pos, 2)
    return w_pos, w_neg

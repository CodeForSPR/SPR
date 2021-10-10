from __future__ import print_function

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from random import randint
import numpy as np
import torch
import torch.nn as nn
import warnings

from encoders import LayerEncoder
from aggregators import FirstLayerAggregator, NonFirstLayerAggregator

warnings.filterwarnings("ignore")


class SPR(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_feats,
                 features,
                 train_pos=None,
                 train_neg=None,
                 train_pos_dir=None,
                 train_neg_dir=None,
                 test_pos_dir=None,
                 test_neg_dir=None,
                 cuda_available=False,
                 alpha=0.2,
                 dropout=0.5,
                 nheads=4,
                 class_weights=None,
                 criterion_lambda=None,
                 criterion_gamma=None,
                 hidden_dim=64
                 ):
        super(SPR, self).__init__()
        self.num_nodes = num_nodes
        self.train_pos, self.train_neg = train_pos, train_neg
        self.train_pos_dir, self.train_neg_dir = train_pos_dir, train_neg_dir
        self.test_pos_dir, self.test_neg_dir = test_pos_dir, test_neg_dir

        self.features = features
        self.cuda_available = cuda_available

        self.criterion_gamma = criterion_gamma
        self.criterion_lambda = criterion_lambda

        layer1_in_dim, layer1_out_dim, nheads1 = num_feats, hidden_dim, nheads
        layer2_in_dim, layer2_out_dim, nheads2 = layer1_out_dim * nheads1, hidden_dim, nheads
        final_out_dim = hidden_dim

        self.edge_type_embedding = nn.Parameter(torch.Tensor(2, hidden_dim))
        nn.init.xavier_normal_(self.edge_type_embedding)
        # -----------------------------------------------
        self.agg1 = FirstLayerAggregator(1, features, only_layer=False, cuda=cuda_available,
                                         in_feat_dim=layer1_in_dim, out_feat_dim=layer1_out_dim, nheads=nheads1,
                                         alpha=alpha, dropout=dropout, edge_embedding=self.edge_type_embedding)
        self.enc1 = LayerEncoder(1, train_pos, train_neg, self.agg1, base_model=None, last_layer=False)
        self.agg2 = NonFirstLayerAggregator(2, lambda nodes: self.enc1(nodes), cuda=cuda_available,
                                            in_feat_dim=layer2_in_dim, out_feat_dim=layer2_out_dim, nheads=nheads2,
                                            alpha=alpha, dropout=dropout)
        self.enc = LayerEncoder(2, train_pos, train_neg, self.agg2, base_model=self.enc1, last_layer=True)
        # -----------------------------------------------
        self.nllloss_func = nn.NLLLoss(weight=torch.FloatTensor(class_weights))
        self.transform = nn.Bilinear(final_out_dim, final_out_dim, 1, bias=True)
        self.structural_distance = nn.PairwiseDistance(p=2)

    def forward(self, nodes):
        node_embedding = self.enc(nodes)
        return node_embedding

    def loss(self, center_nodes):
        max_node_index = self.num_nodes - 1
        i_loss2, pos_no_loss2, no_neg_loss2 = [], [], []
        i_indices, j_indices, ys = [], [], []
        all_nodes_set, skipped_nodes = set(), []

        for i in center_nodes:
            if (len(self.train_pos[i]) + len(self.train_neg[i])) == 0:
                skipped_nodes.append(i)
                continue
            all_nodes_set.add(i)
            for j_pos in self.train_pos[i]:
                i_loss2.append(i)
                pos_no_loss2.append(j_pos)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in self.train_pos[i]) and (temp not in self.train_neg[i]):
                        break
                no_neg_loss2.append(temp)
                all_nodes_set.add(temp)
                all_nodes_set.add(j_pos)

                i_indices.append(i)
                j_indices.append(j_pos)
                ys.append(0)

            for j_neg in self.train_neg[i]:
                i_loss2.append(i)
                no_neg_loss2.append(j_neg)
                while True:
                    temp = randint(0, max_node_index)
                    if (temp not in self.train_pos[i]) and (temp not in self.train_neg[i]):
                        break
                pos_no_loss2.append(temp)
                all_nodes_set.add(temp)
                all_nodes_set.add(j_neg)

                i_indices.append(i)
                j_indices.append(j_neg)
                ys.append(1)

        all_nodes_list = sorted(list(all_nodes_set))
        all_nodes_map = {node: i for i, node in enumerate(all_nodes_list)}
        final_embedding, edge_type_embedding = self.forward(all_nodes_list)

        i_indices_mapped = [all_nodes_map[i] for i in i_indices]
        j_indices_mapped = [all_nodes_map[j] for j in j_indices]

        ys = torch.LongTensor(ys)
        if self.cuda_available:
            ys = ys.cuda()

        # boundary classifier
        positive_prob = torch.sigmoid(self.transform(final_embedding[i_indices_mapped], final_embedding[j_indices_mapped]))
        negative_prob = 1 - positive_prob
        probs = torch.cat((positive_prob, negative_prob), dim=1)
        probs = torch.clamp(probs, 1e-10, 1e10)
        probs_log = torch.log(probs)
        loss_separable = self.nllloss_func(probs_log, ys)
        # --------------------------------------------------------------------------------
        i_loss2 = [all_nodes_map[i] for i in i_loss2]
        pos_no_loss2 = [all_nodes_map[i] for i in pos_no_loss2]
        no_neg_loss2 = [all_nodes_map[i] for i in no_neg_loss2]
        tensor_zeros = torch.zeros(len(i_loss2))
        if self.cuda_available:
            tensor_zeros = tensor_zeros.cuda()
        loss_structure = torch.mean(
            torch.max(tensor_zeros,
                      self.structural_distance(final_embedding[i_loss2], final_embedding[pos_no_loss2]) ** 2
                      - self.structural_distance(final_embedding[i_loss2], final_embedding[no_neg_loss2]) ** 2
                      ))
        # --------------------------------------------------------------------------------
        idx_select = lambda x, idx: torch.LongTensor(np.argwhere(x.numpy() == idx)).view(1, -1)[0]
        indices0 = idx_select(ys.cpu(), 0) if self.cuda_available else idx_select(ys, 0)
        indices1 = idx_select(ys.cpu(), 1) if self.cuda_available else idx_select(ys, 1)
        #
        i_loss2 = torch.LongTensor(i_loss2)
        pos_no_loss2 = torch.LongTensor(pos_no_loss2)
        no_neg_loss2 = torch.LongTensor(no_neg_loss2)

        pos_indices = i_loss2[indices0]
        neg_indices = i_loss2[indices1]
        pos_connected = pos_no_loss2[indices0]
        pos_unconnected = no_neg_loss2[indices0]
        neg_connected = no_neg_loss2[indices1]
        neg_unconnected = pos_no_loss2[indices1]
        #
        scores = lambda i1, i2, idx: torch.sigmoid(torch.sum(final_embedding[i1] * final_embedding[i2] * edge_type_embedding[idx], dim=1))

        s_pos = scores(pos_indices, pos_connected, 0) * (1 - scores(pos_indices, pos_connected, 1))
        s_neg = (1 - scores(neg_indices, neg_connected, 0)) * scores(neg_indices, neg_connected, 1)
        s_nul = torch.cat(((1 - scores(pos_indices, pos_unconnected, 0)) * (1 - scores(pos_indices, pos_unconnected, 1)),
                           (1 - scores(neg_indices, neg_unconnected, 0)) * (1 - scores(neg_indices, neg_unconnected, 1))), dim=0)
        loss_proximity = torch.mean(-1 * torch.log(torch.clamp(torch.cat((s_pos, s_neg, s_nul), dim=0), 1e-10, 1e10)))
        # loss_proximity = torch.mean(-1 * torch.log(torch.cat((s_pos, s_neg, s_nul), dim=0)))
        # --------------------------------------------------------------------------------

        return loss_separable + self.criterion_gamma * loss_proximity + self.criterion_lambda * loss_structure

    def test_func(self, last_epoch=False, path=None):
        all_nodes_list = list(range(self.num_nodes))
        final_embedding, _ = self.forward(all_nodes_list)
        if self.cuda_available:
            final_embedding = final_embedding.detach().cpu().numpy()
        else:
            final_embedding = final_embedding.detach().numpy()

        # ---------------------------------------------------
        X_train = []
        y_train = []
        X_val = []
        y_test_true = []
        # ---------------------------------------------------

        for i in range(self.num_nodes):
            for j in self.train_pos_dir[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(1)

            for j in self.train_neg_dir[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_train.append(temp)
                y_train.append(0)

            for j in self.test_pos_dir[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(1)

            for j in self.test_neg_dir[i]:
                temp = np.append(final_embedding[i], final_embedding[j])
                X_val.append(temp)
                y_test_true.append(0)

        train_X1 = np.array(X_train)
        test_X1 = np.array(X_val)
        train_y = np.array([int((i + 1) / 2) for i in y_train])
        test_y = np.array([int((i + 1) / 2) for i in y_test_true])

        logistic_function = LogisticRegression()
        logistic_function.fit(train_X1, train_y)

        pred = logistic_function.predict(test_X1)
        pred_p = logistic_function.predict_proba(test_X1)

        f1_macro = metrics.f1_score(test_y, pred, average='macro')
        f1_micro = metrics.f1_score(test_y, pred, average='micro')
        f1_weighted = metrics.f1_score(test_y, pred, average='weighted')
        f1_binary = metrics.f1_score(test_y, pred, average='binary')
        auc = metrics.roc_auc_score(test_y, pred_p[:, 1])

        print(metrics.confusion_matrix(test_y, pred))
        print('f1_mi', f1_micro,
              'f1_ma', f1_macro,
              'f1_wt', f1_weighted,
              'f1_bi', f1_binary,
              'auc', auc,
              )

        if last_epoch and path is not None:
            np.savetxt(path, final_embedding)

        return round(f1_micro, 4), round(f1_macro, 4), round(f1_weighted, 4), round(f1_binary, 4), round(auc, 4)

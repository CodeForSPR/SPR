import torch
import torch.nn as nn

from layers import SpMergeAttentionLayer


class NonFirstLayerAggregator(nn.Module):
    def __init__(self, _id, node_features, cuda=False, in_feat_dim=64, out_feat_dim=64,
                 nheads=4, alpha=0.2, dropout=0.5):
        super(NonFirstLayerAggregator, self).__init__()
        self.id = _id
        self.node_features = node_features
        self.cuda = cuda
        self.act_func = nn.ReLU()

        self.attentions = [SpMergeAttentionLayer(in_features_node=in_feat_dim, out_features_node=out_feat_dim, cuda_available=cuda,
                                                 alpha=alpha, dropout=dropout, layer_id=2) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}_{}'.format(self.id, i), attention)

    def forward(self, nodes, neighs_pos, neighs_neg):
        self_nodes = [{nodes[i]} for i, neigh in enumerate(nodes)]
        unique_nodes_list = sorted(list(set.union(*neighs_pos).union(*neighs_neg).union(*self_nodes)))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # col indices
        col_indices_pos = [unique_nodes[n] for neigh in neighs_pos for n in neigh]
        col_indices_neg = [unique_nodes[n] for neigh in neighs_neg for n in neigh]
        col_indices_self = [unique_nodes[selfnode] for selfset in self_nodes for selfnode in selfset]
        # row indices
        row_indices_pos = [i for i in range(len(neighs_pos)) for _ in range(len(neighs_pos[i]))]
        row_indices_neg = [i for i in range(len(neighs_neg)) for _ in range(len(neighs_neg[i]))]
        row_indices_self = [i for i in range(len(self_nodes)) for _ in range(len(self_nodes[i]))]

        adj_pos = torch.tensor([row_indices_pos, col_indices_pos])
        adj_neg = torch.tensor([row_indices_neg, col_indices_neg])
        adj_self = torch.tensor([row_indices_self, col_indices_self])
        adj_pos2 = torch.cat((adj_pos, adj_self), dim=1)

        if self.cuda:
            adj_pos2 = adj_pos2.cuda()
            adj_neg = adj_neg.cuda()
            embed_matrix = self.node_features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.node_features(torch.LongTensor(unique_nodes_list))
        node_features, edge_embedding = embed_matrix

        if len(self.attentions) > 1:
            h_hidden = 0
            for attlayer in self.attentions:
                h_hidden += attlayer(node_features, adj_pos2, adj_neg, shape=(len(self_nodes), len(unique_nodes)),
                                     edge_embedding=edge_embedding)
            h_hidden /= len(self.attentions)
        else:
            h_hidden = self.attentions[0](
                node_features, adj_pos2, adj_neg, shape=(len(self_nodes), len(unique_nodes)), edge_embedding=edge_embedding)

        h_hidden = self.act_func(h_hidden)
        return h_hidden, edge_embedding


class FirstLayerAggregator(nn.Module):
    def __init__(self, _id, node_features, only_layer, cuda=False, in_feat_dim=64, out_feat_dim=64,
                 nheads=4, alpha=0.2, dropout=0.5, edge_embedding=None, edge_embedding_dim=64):
        super(FirstLayerAggregator, self).__init__()
        self.id = _id
        self.node_features = node_features
        self.only_layer = only_layer
        self.cuda = cuda
        self.act_func = nn.ReLU()

        self.edge_embedding = edge_embedding
        self.edge_transform = nn.Parameter(torch.Tensor(edge_embedding_dim, out_feat_dim))
        nn.init.xavier_normal_(self.edge_transform.data)

        self.attentions = [SpMergeAttentionLayer(in_features_node=in_feat_dim, out_features_node=out_feat_dim, cuda_available=cuda,
                                                 alpha=alpha, dropout=dropout, layer_id=1) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}_{}'.format(self.id, i), attention)

    def forward(self, nodes, neighs_pos, neighs_neg):
        if self.only_layer:
            self_nodes = [{nodes[i]} for i, neigh in enumerate(nodes)]
        else:
            self_nodes = [{nodes[i].item()} for i, neigh in enumerate(nodes)]

        unique_nodes_list = sorted(list(set.union(*neighs_pos).union(*neighs_neg).union(*self_nodes)))
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # col indices
        col_indices_pos = [unique_nodes[n] for neigh in neighs_pos for n in sorted(neigh)]
        col_indices_neg = [unique_nodes[n] for neigh in neighs_neg for n in sorted(neigh)]
        col_indices_self = [unique_nodes[selfnode] for selfset in self_nodes for selfnode in selfset]
        # row indices
        row_indices_pos = [i for i in range(len(neighs_pos)) for _ in range(len(neighs_pos[i]))]
        row_indices_neg = [i for i in range(len(neighs_neg)) for _ in range(len(neighs_neg[i]))]
        row_indices_self = [i for i in range(len(self_nodes)) for _ in range(len(self_nodes[i]))]

        adj_pos = torch.tensor([row_indices_pos, col_indices_pos])
        adj_neg = torch.tensor([row_indices_neg, col_indices_neg])
        adj_self = torch.tensor([row_indices_self, col_indices_self])
        adj_pos2 = torch.cat((adj_pos, adj_self), dim=1)

        if self.cuda:
            adj_pos2 = adj_pos2.cuda()
            adj_neg = adj_neg.cuda()
            embed_matrix = self.node_features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.node_features(torch.LongTensor(unique_nodes_list))

        if len(self.attentions) > 1:
            h_hidden = torch.cat([
                att(embed_matrix, adj_pos2, adj_neg, shape=(len(self_nodes), len(unique_nodes)), edge_embedding=self.edge_embedding)
                for att in self.attentions], dim=1)
        else:
            h_hidden = self.attentions[0](
                embed_matrix, adj_pos2, adj_neg, shape=(len(self_nodes), len(unique_nodes)), edge_embedding=self.edge_embedding)

        h_hidden = self.act_func(h_hidden)
        edge_embedding = torch.matmul(self.edge_embedding, self.edge_transform)
        return h_hidden, edge_embedding

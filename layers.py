import torch
import torch.nn as nn


class SpecialSpmmFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad is False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    @staticmethod
    def forward(indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpMergeAttentionLayer(nn.Module):
    def __init__(self, in_features_node,
                 out_features_node,
                 cuda_available=False,
                 alpha=0.2,
                 num_relations=2,
                 bias=True,
                 dropout=0.5,
                 layer_id=None
                 ):
        super(SpMergeAttentionLayer, self).__init__()
        self.in_features_node = in_features_node
        self.out_features_node = out_features_node
        self.cuda_available = cuda_available
        self.id = layer_id

        if bias:
            self.bias = nn.Parameter(torch.zeros(size=(1, out_features_node)))
            nn.init.xavier_normal_(self.bias.data)
            self.add_bias = True
        else:
            self.add_bias = False
        self.Wr = nn.Parameter(torch.Tensor(num_relations, in_features_node, out_features_node))
        self.a = nn.Parameter(torch.zeros(size=(1, 2 * out_features_node)))
        nn.init.xavier_normal_(self.a.data)
        nn.init.xavier_normal_(self.Wr.data, gain=nn.init.calculate_gain('relu'))

        self.act_func = nn.LeakyReLU(alpha)
        self.spmm = SpecialSpmm()
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_embedding, adj_pos, adj_neg, shape=None, edge_embedding=None):
        if shape is None:
            N = node_embedding.size()[0]
            M = N
        else:
            N, M = shape

        num_pos, num_neg = adj_pos.size()[1], adj_neg.size()[1]

        h_pos, h_neg = torch.mm(node_embedding, self.Wr[0]), torch.mm(node_embedding, self.Wr[1])
        h_pos, h_neg = torch.mm(h_pos, torch.diag(edge_embedding[0])), torch.mm(h_neg, torch.diag(edge_embedding[1]))

        edge_pos = torch.cat((h_pos[adj_pos[0, :], :], h_pos[adj_pos[1, :], :]), dim=1)
        edge_neg = torch.cat((h_neg[adj_neg[0, :], :], h_neg[adj_neg[1, :], :]), dim=1)
        edge_all = torch.cat((edge_pos, edge_neg), dim=0)

        edge_e = torch.sigmoid(self.act_func(self.a.mm(edge_all.t()).squeeze()))

        tensor_ones_M = torch.ones(size=(M, 1))
        if self.cuda_available:
            tensor_ones_M = tensor_ones_M.cuda()
        edge = torch.cat((adj_pos, adj_neg), dim=1)
        e_row_sum = self.spmm(edge, edge_e, torch.Size([N, M]), tensor_ones_M) + 1e-8
        edge_e = self.dropout(edge_e)

        h_agg_pos = self.spmm(adj_pos, edge_e[:num_pos], torch.Size([N, M]), h_pos)
        h_agg_neg = self.spmm(adj_neg, edge_e[-num_neg:], torch.Size([N, M]), h_neg)

        h_agg = h_agg_pos + h_agg_neg
        output = h_agg.div(e_row_sum)

        if self.add_bias:
            output = output + self.bias
        return output

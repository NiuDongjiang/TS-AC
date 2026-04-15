import torch.nn as nn
from dgllife.model.model_zoo.gcn_predictor import MLPPredictor
from dgllife.model.gnn.gcn import GCN
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv,TransformerConv
import torch.nn.functional as F
import torch

class InterGraphAttention(nn.Module):
    def __init__(self, input_dim, dp, head, head_out_feats):
        super().__init__()
        self.input_dim = input_dim
        self.inter = GATConv((input_dim, input_dim), head_out_feats // 2, head, dropout=dp)
        # self.inter = TransformerConv((input_dim, input_dim), head_out_feats // 2, head, dropout=dp)

    def forward(self, h_data, t_data, b_graph):
        edge_index = b_graph.edge_index
        h_input = F.relu(h_data)
        t_input = F.relu(t_data)
        t_rep = self.inter((h_input, t_input), edge_index)
        h_rep = self.inter((t_input, h_input), edge_index[[1, 0]])
        return h_rep, t_rep

class GCNPredictor(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNPredictor, self).__init__()
        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=[int(num_feats) for num_feats in hidden_feats],
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)

        self.gnn1 = GCN(in_feats=int(hidden_feats[0]),
                        hidden_feats=[int(num_feats) * 2 for num_feats in hidden_feats],
                        activation=activation,
                        residual=residual,
                        batchnorm=batchnorm,
                        dropout=dropout)

        self.gnn2 = GCN(in_feats=int(hidden_feats[0]) * 2,
                        hidden_feats=[int(num_feats) * 4 for num_feats in hidden_feats],
                        activation=activation,
                        residual=residual,
                        batchnorm=batchnorm,
                        dropout=dropout)

        gnn_out_feats = self.gnn2.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats):

        node_feats = self.gnn(bg, feats)
        bg.ndata['x'] = node_feats.double()
        node_feats1 = self.gnn1(bg, node_feats)
        bg.ndata['x'] = node_feats1.double()
        node_feats2 = self.gnn2(bg, node_feats1)
        graph_feats = self.readout(bg, node_feats2)
        return node_feats2


class GCNPredictor_mmp(nn.Module):

    def __init__(self, in_feats, hidden_feats=None, gnn_norm=None, activation=None,
                 residual=None, batchnorm=None, dropout=None, classifier_hidden_feats=128,
                 classifier_dropout=0., n_tasks=1, predictor_hidden_feats=128,
                 predictor_dropout=0.):
        super(GCNPredictor_mmp, self).__init__()
        if predictor_hidden_feats == 128 and classifier_hidden_feats != 128:
            print('classifier_hidden_feats is deprecated and will be removed in the future, '
                  'use predictor_hidden_feats instead')
            predictor_hidden_feats = classifier_hidden_feats

        if predictor_dropout == 0. and classifier_dropout != 0.:
            print('classifier_dropout is deprecated and will be removed in the future, '
                  'use predictor_dropout instead')
            predictor_dropout = classifier_dropout

        self.gnn = GCN(in_feats=in_feats,
                       hidden_feats=[int(num_feats) for num_feats in hidden_feats],
                       activation=activation,
                       residual=residual,
                       batchnorm=batchnorm,
                       dropout=dropout)

        self.gnn1 = GCN(in_feats=int(hidden_feats[0]),
                        hidden_feats=[int(num_feats) * 2 for num_feats in hidden_feats],
                        activation=activation,
                        residual=residual,
                        batchnorm=batchnorm,
                        dropout=dropout)

        self.gnn2 = GCN(in_feats=int(hidden_feats[0]) * 2,
                        hidden_feats=[int(num_feats) * 4 for num_feats in hidden_feats],
                        activation=activation,
                        residual=residual,
                        batchnorm=batchnorm,
                        dropout=dropout)

        gnn_out_feats = self.gnn2.hidden_feats[-1]
        self.readout = WeightedSumAndMax(gnn_out_feats)
        self.predict = MLPPredictor(2 * gnn_out_feats, predictor_hidden_feats,
                                    n_tasks, predictor_dropout)

    def forward(self, bg, feats, pyg, i, t, no):
        node_feats = self.gnn(bg, feats)
        bg.ndata['x'] = node_feats.double()
        node_feats1 = self.gnn1(bg, node_feats)
        bg.ndata['x'] = node_feats1.double()
        node_feats2 = self.gnn2(bg, node_feats1)
        if t == 'train':
            torch.save(node_feats2, 'mmp_molregno{}_{}.pth'.format(no, i))
            torch.save(pyg, 'mmp_pyg{}_ID_{}.pth'.format(no, i))
        graph_feats = self.readout(bg, node_feats2)
        return self.predict(graph_feats)

import torch
import torch.nn as nn
from utils.GCNPredictor import GCNPredictor
from utils.KANLayer import KANLayer
from torch_geometric.nn import GCNConv,SAGPooling,global_add_pool,GATConv,TransformerConv
import torch.nn.functional as F
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax


class GatedMultimodalLayer(nn.Module):
    """
    Gated Multimodal Layer based on 'Gated multimodal networks,
    Arevalo1 et al.' (https://arxiv.org/abs/1702.01992)
    """

    def __init__(self, size_in1, size_in2, size_out=16):
        super(GatedMultimodalLayer, self).__init__()
        self.size_in1, self.size_in2, self.size_out = size_in1, size_in2, size_out

        self.hidden1 = nn.Linear(size_in1, size_out, bias=False)
        self.hidden2 = nn.Linear(size_in2, size_out, bias=False)
        self.hidden_sigmoid = nn.Linear(size_out * 2, 1, bias=False)

        # Activation functions
        self.tanh_f = nn.ReLU()
        self.sigmoid_f = nn.Sigmoid()

    def forward(self, x1, x2):
        h1 = self.tanh_f(self.hidden1(x1))
        h2 = self.tanh_f(self.hidden2(x2))
        x = torch.cat((h1, h2), dim=1)
        z = self.sigmoid_f(self.hidden_sigmoid(x))

        return z.view(z.size()[0], 1) * h1 + (1 - z).view(z.size()[0], 1) * h2

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

class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta

class TS_SUB(nn.Module):

    def __init__(self, args):
        super(TS_SUB, self).__init__()
        
        number_atom_features = 32
        graph_conv_layers_core: list = [64, 64]
        graph_conv_layers_sub: list = [64, 64]
        activation = None
        residual: bool = True
        batchnorm: bool = True
        dropout: float = args['DROP_OUT']
        out_size_core = 64
        out_size_sub = 64
        predictor_hidden_feats_core: int = 64
        predictor_hidden_feats_sub: int = 64
        predictor_dropout: float = args['DROP_OUT']
        num_gnn_layers = len(graph_conv_layers_core)

        self.gcn_layer_core = GCNPredictor(
                            in_feats=number_atom_features,
                            hidden_feats=graph_conv_layers_core,
                            activation=activation,
                            residual=[residual] * num_gnn_layers,
                            batchnorm=[batchnorm] * num_gnn_layers,
                            dropout=[dropout] * num_gnn_layers,
                            n_tasks=out_size_core,
                            predictor_hidden_feats=predictor_hidden_feats_core,
                            predictor_dropout=predictor_dropout)
        self.gcn_layer_sub = GCNPredictor(
            in_feats=number_atom_features,
            hidden_feats=graph_conv_layers_sub,
            activation=activation,
            residual=[residual] * num_gnn_layers,
            batchnorm=[batchnorm] * num_gnn_layers,
            dropout=[dropout] * num_gnn_layers,
            n_tasks=out_size_sub,
            predictor_hidden_feats=predictor_hidden_feats_sub,
            predictor_dropout=predictor_dropout)

        self.interAtt = InterGraphAttention(256, args['DROP_OUT'], args["tr_head"], 256)
        '''
        self.fc_layer2 = nn.Sequential(nn.Linear(1024, 512),
                           nn.BatchNorm1d(512),
                           nn.ReLU(inplace=True),
                           nn.Dropout(args['DROP_OUT']),
                           nn.Linear(512, 256),
                           nn.BatchNorm1d(256),
                           nn.ReLU(inplace=True),
                           nn.Dropout(args['DROP_OUT']))
        '''
        self.fc_layer2 = nn.Sequential(nn.Linear(1024 * 2, 1024),
                                       nn.BatchNorm1d(1024),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(args['DROP_OUT']),
                                       nn.Linear(1024, 512),
                                       nn.BatchNorm1d(512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(args['DROP_OUT']))


        self.atom_fc = nn.Linear(32,64)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=args['tr_head'])
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args['tr_layer'])
        self.p_f1 = nn.Linear(1, 1024)
        self.p_f2 = nn.Linear(1, 1024)
        self.delta_f = nn.Linear(1, 2048)
        self.attention = Attention(1024)

        self.out_layer = nn.Linear(512, 1)
        self.kan_layer1 = KANLayer(in_dim=1024*2, out_dim=1)
        self.gatedfusion = GatedMultimodalLayer(1024, 1024, 1024)

    def forward(self, batch_core, batch_sub1, batch_sub2, b1, b2, sub1_pyg, sub2_pyg, i, t):
        core = self.gcn_layer_core(batch_core, batch_core.ndata['x'].float())
        sub1 = self.gcn_layer_sub(batch_sub1, batch_sub1.ndata['x'].float())
        sub2 = self.gcn_layer_sub(batch_sub2, batch_sub2.ndata['x'].float())

        core1, sub1 = self.interAtt(core, sub1, b1)
        core2, sub2 = self.interAtt(core, sub2, b2)
        if t == 'train':
            torch.save(core1, 'sub_core1_{}.pth'.format(i))
            torch.save(core2, 'sub_core2_{}.pth'.format(i))
            torch.save(sub1, 'sub_sub1_{}.pth'.format(i))
            torch.save(sub2, 'sub_sub2_{}.pth'.format(i))
            torch.save(sub1_pyg, 'sub_pyg1_ID_{}.pth'.format(i))
            torch.save(sub2_pyg, 'sub_pyg2_ID_{}.pth'.format(i))
            torch.save(b1, 'sub_core_pyg_{}.pth'.format(i))

        core1 = global_add_pool(core1, b1.batch)
        core2 = global_add_pool(core2, b2.batch)
        #core1 = global_add_pool(core, b1.batch)
        #core2 = global_add_pool(core, b2.batch)
        sub1 = global_add_pool(sub1, sub1_pyg.batch)
        sub2 = global_add_pool(sub2, sub2_pyg.batch)

        s1 = torch.cat([core1, sub1], 1)
        s2 = torch.cat([core2, sub2], 1)

        #p1 = sub1_pyg.p.unsqueeze(1)
        #p2 = sub2_pyg.p.unsqueeze(1)
        #delta = torch.abs(p1 - p2)
        #delta = self.delta_f(delta)

        p1 = self.p_f1(sub1_pyg.p.unsqueeze(1))
        p2 = self.p_f2(sub2_pyg.p.unsqueeze(1))
        '''
        s1 = self.gatedfusion(s1, p1)
        s2 = self.gatedfusion(s2, p2)
        
        s1 = torch.stack([s1, p1], dim=1)
        s1, att_1= self.attention(s1)

        s2 = torch.stack([s2, p2], dim=1)
        s2, att_2 = self.attention(s2)
        '''
        drug_out = torch.cat([s1, s2], 1)

        out = self.fc_layer2(drug_out)
        out = self.out_layer(out)
        #out, preacts, postacts, postspline = self.out_layer1(out)
        out = torch.sigmoid(out)
        out = out.squeeze(-1)

        return out, s1, p1, s2, p2

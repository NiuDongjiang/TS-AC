import torch
import torch.nn as nn
from utils.GCNPredictor import GCNPredictor_mmp
from utils.KANLayer import KANLayer
from torch_geometric.utils import degree

class TS_MMP(nn.Module):

    def __init__(self, args):
        super(TS_MMP, self).__init__()

        number_atom_features = 32
        graph_conv_layers: list = [128, 128]
        activation = None
        residual: bool = True
        batchnorm: bool = True
        dropout: float = args['DROP_OUT']#0.5 ###
        out_size = 128*2
        predictor_hidden_feats: int = 128*2
        predictor_dropout: float = args['DROP_OUT'] ###
        num_gnn_layers = len(graph_conv_layers)

        self.gcn_layer = GCNPredictor_mmp(in_feats=number_atom_features,
                                    hidden_feats=graph_conv_layers,
                                    activation=activation,
                                    residual=[residual] * num_gnn_layers,
                                    batchnorm=[batchnorm] * num_gnn_layers,
                                    dropout=[dropout] * num_gnn_layers,
                                    n_tasks=out_size,
                                    predictor_hidden_feats=predictor_hidden_feats,
                                    predictor_dropout=predictor_dropout)

        self.fc_layer2 = nn.Sequential(nn.Linear(128*4, 128*2),
                                           nn.BatchNorm1d(128*2),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(args['DROP_OUT']),nn.Linear(128*2, 128*1),
                                           nn.BatchNorm1d(128*1),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(args['DROP_OUT'])) ###

        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=args['tr_head'])
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args['tr_layer'])

        self.out_layer = nn.Linear(128*1, 1)
        self.out_layer1 = KANLayer(in_dim=128*4,out_dim=1)

    def forward(self, batch_smiles1, batch_smiles2, pyg1, pyg2, i, t):
        smiles1 = self.gcn_layer(batch_smiles1, batch_smiles1.ndata['x'].float(), pyg1, i, t, no=1)
        smiles2 = self.gcn_layer(batch_smiles2, batch_smiles2.ndata['x'].float(), pyg2, i, t, no=2)

        drug_out = torch.stack((smiles1, smiles2), dim=1)
        drug_out = self.drug_trans(drug_out)
        drug_out = drug_out.view(drug_out.shape[0], 2 * drug_out.shape[-1])
        #out = torch.cat((smiles1, smiles2), axis=1)
        out = self.fc_layer2(drug_out)
        out = self.out_layer(out)
        #out, preacts, postacts, postspline = self.out_layer1(drug_out)
        out = torch.sigmoid(out)
        out = out.squeeze(-1)
        
        return out
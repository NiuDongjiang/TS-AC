import dgl
import torch
import numpy as np
from rdkit import Chem
from torch_geometric.data import Data


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return one_of_k_encoding(atom.GetSymbol(),
                             ['Br', 'C', 'Cl', 'F', 'H', 'I', 'N', 'O', 'P', 'S', 'B', 'Unknown']) \
           + one_of_k_encoding(atom.GetDegree(), list(range(7))) \
           + one_of_k_encoding(atom.GetImplicitValence(), list(range(7))) \
           + one_of_k_encoding(atom.GetHybridization(), [
            Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, Chem.rdchem.HybridizationType.SP3D2]) \
           + [atom.GetIsAromatic()]


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()], dtype=np.float)


def create_graph(smiles, device, molregno):

    mol = Chem.MolFromSmiles(smiles)

    # Extract atom features (node features)
    atoms = mol.GetAtoms()
    atom_tensor = np.zeros((len(atoms), 32))
    for atoms_ix, atom in enumerate(atoms):
        atom_tensor[atoms_ix, :] = atom_features(atom)
    atom_tensor = torch.from_numpy(atom_tensor).double()

    # Extract bond features (edge features)
    bonds = mol.GetBonds()
    bond_tensor = []
    for bond in bonds:
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = np.array(bond_features(bond))
        bond_tensor.append([begin_idx, end_idx, features])
        bond_tensor.append([end_idx, begin_idx, features])
    b = Data(x=atom_tensor, molregno=molregno)

    bond_tensor.sort()

    bond_idx1 = [t[0] for t in bond_tensor]
    bond_idx2 = [t[1] for t in bond_tensor]
    bond_tensor = [t[2] for t in bond_tensor]
    bond_tensor = torch.from_numpy(np.array(bond_tensor)).double()

    G = dgl.DGLGraph().to(device)

    # Add N nodes
    G.add_nodes(len(atom_tensor))
    # Add edges
    G.add_edges(bond_idx1, bond_idx2)

    # Add node features
    G.ndata['x'] = atom_tensor.to(device)
    # Add edge features
    G.edata['y'] = bond_tensor.to(device)
    G = dgl.add_self_loop(G)

    return G,b

def create_graph_sub(smiles, p, device, molregno):

    mol = Chem.MolFromSmiles(smiles)

    # Extract atom features (node features)
    atoms = mol.GetAtoms()
    atom_tensor = np.zeros((len(atoms), 32))
    for atoms_ix, atom in enumerate(atoms):
        atom_tensor[atoms_ix, :] = atom_features(atom)
    atom_tensor = torch.from_numpy(atom_tensor).double()

    # Extract bond features (edge features)
    bonds = mol.GetBonds()
    bond_tensor = []
    for bond in bonds:
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = np.array(bond_features(bond))
        bond_tensor.append([begin_idx, end_idx, features])
        bond_tensor.append([end_idx, begin_idx, features])
    b = Data(x=atom_tensor, p=p, molregno=molregno)

    bond_tensor.sort()

    bond_idx1 = [t[0] for t in bond_tensor]
    bond_idx2 = [t[1] for t in bond_tensor]
    bond_tensor = [t[2] for t in bond_tensor]
    bond_tensor = torch.from_numpy(np.array(bond_tensor)).double()

    G = dgl.DGLGraph().to(device)

    # Add N nodes
    G.add_nodes(len(atom_tensor))
    # Add edges
    G.add_edges(bond_idx1, bond_idx2)

    # Add node features
    G.ndata['x'] = atom_tensor.to(device)
    # Add edge features
    G.edata['y'] = bond_tensor.to(device)
    G = dgl.add_self_loop(G)

    return G,b

def create_graph_core(smiles, sub1, sub2, device):

    mol = Chem.MolFromSmiles(smiles)
    sub1 = Chem.MolFromSmiles(sub1)
    sub2 = Chem.MolFromSmiles(sub2)

    # Extract atom features (node features)
    atoms = mol.GetAtoms()
    atoms_sub1 = sub1.GetAtoms()
    atoms_sub2 = sub2.GetAtoms()
    atom_tensor = np.zeros((len(atoms), 32))
    atom_tensor_sub1 = np.zeros((len(atoms_sub1), 32))
    atom_tensor_sub2 = np.zeros((len(atoms_sub2), 32))
    for atoms_ix, atom in enumerate(atoms):
        atom_tensor[atoms_ix, :] = atom_features(atom)
    for atoms_ix, atom in enumerate(atoms_sub1):
        atom_tensor_sub1[atoms_ix, :] = atom_features(atom)
    for atoms_ix, atom in enumerate(atoms_sub2):
        atom_tensor_sub2[atoms_ix, :] = atom_features(atom)
    atom_tensor = torch.from_numpy(atom_tensor).double()
    atom_tensor_sub1 = torch.from_numpy(atom_tensor_sub1).double()
    atom_tensor_sub2 = torch.from_numpy(atom_tensor_sub2).double()

    b_graph1 = _create_b_graph(get_bipartite_graph(mol, sub1), atom_tensor, atom_tensor_sub1)
    b_graph2 = _create_b_graph(get_bipartite_graph(mol, sub2), atom_tensor, atom_tensor_sub2)
    #b_graph1 = Data(edge_index=b_graph1.edge_index)
    #b_graph2 = Data(edge_index=b_graph2.edge_index)

    # Extract bond features (edge features)
    bonds = mol.GetBonds()
    bond_tensor = []
    for bond in bonds:
        begin_idx = bond.GetBeginAtom().GetIdx()
        end_idx = bond.GetEndAtom().GetIdx()
        features = np.array(bond_features(bond))
        bond_tensor.append([begin_idx, end_idx, features])
        bond_tensor.append([end_idx, begin_idx, features])

    bond_tensor.sort()

    bond_idx1 = [t[0] for t in bond_tensor]
    bond_idx2 = [t[1] for t in bond_tensor]
    bond_tensor = [t[2] for t in bond_tensor]
    bond_tensor = torch.from_numpy(np.array(bond_tensor)).double()

    G = dgl.DGLGraph().to(device)

    # Add N nodes
    G.add_nodes(len(atom_tensor))
    # Add edges
    G.add_edges(bond_idx1, bond_idx2)

    # Add node features
    G.ndata['x'] = atom_tensor.to(device)
    # Add edge features
    G.edata['y'] = bond_tensor.to(device)

    # Store bipartite graphs as attributes of G
    #setattr(G, 'bipartite_graph1', b_graph1)
    #setattr(G, 'bipartite_graph2', b_graph2)

    return G, b_graph1, b_graph2

class BipartiteData(Data):
    def __init__(self, edge_index=None, x_s=None, x_t=None):
        super().__init__()
        self.edge_index = edge_index
        self.x_s = x_s
        self.x_t = x_t
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        else:
            return super().__inc__(key, value, *args, **kwargs)

def _create_b_graph(edge_index,x_s, x_t):
    return BipartiteData(edge_index,x_s,x_t)

def get_bipartite_graph(mol_graph_1,mol_graph_2):
    x1 = np.arange(0,len(mol_graph_1.GetAtoms()))
    x2 = np.arange(0,len(mol_graph_2.GetAtoms()))
    edge_list = torch.LongTensor(np.meshgrid(x1,x2))
    edge_list = torch.stack([edge_list[0].reshape(-1),edge_list[1].reshape(-1)])
    return edge_list
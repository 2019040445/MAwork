import numpy as np
import torch
from torch.utils import data
from rdkit import Chem
from gen_mol_graph import *
from configs import Model_config
config = Model_config()

import pandas as pd
from rdkit.Chem.rdchem import BondType
from graph_features import atom_features
from collections import defaultdict
from subword_nmt.apply_bpe import BPE
import codecs


def drug_embedding(id):
    x, edge_attr, edge_index = sdf2graph(id)
    N = x.size(0)
    x = mol_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = mol_to_single_emb(edge_attr) + 1

    shortest_path_result, path = algos.floyd_warshall(adj.numpy())
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + 1, N + 1], dtype=torch.float)  # with graph token

    node = x
    attn_bias = attn_bias
    spatial_pos = spatial_pos
    in_degree = adj.long().sum(dim=1).view(-1)
    out_degree = adj.long().sum(dim=0).view(-1)
    edge_input = torch.from_numpy(edge_input).long()

    # print(node.shape) [node_num,9]

    return node, attn_bias, spatial_pos, in_degree, out_degree, edge_input


class Dataset(data.Dataset):

    def __init__(self, list_IDs, labels, df_dti):
        'Initialization'
        self.labels = labels #86
        self.list_IDs = list_IDs #all index
        self.df = df_dti #index drug_ids labels
        self.drug1_id = self.df["Drug1"].values #all id1 'numpy.ndarray'
        self.drug2_id = self.df["Drug2"].values
        # self.smiles1 = np.array(['CC1=NC=C(COP(O)(O)=O)C(C=O)=C1O','CC1=NC=C(COP(O)(O)=O)C(C=O)=C1O'])
        # self.smiles2 = np.array(['CC1=NC=C(COP(O)(O)=O)C(C=O)=C1O','CC1=NC=C(COP(O)(O)=O)C(C=O)=C1O'])
        self.smiles1 =np.array(get_smiles(self.drug1_id)) #all smiles1
        self.smiles2 =np.array(get_smiles(self.drug2_id))

        
    def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)



    def __getitem__(self, index):
        '''AMDE-dataset'''
        index = self.list_IDs[index]
        adj_1, nd_1, ed_1 = smile_to_graph(self.smiles1[index])
        adj_2, nd_2, ed_2 = smile_to_graph(self.smiles2[index])

        d1, mask_1 = drug2emb_encoder(self.smiles1[index])
        d2, mask_2 = drug2emb_encoder(self.smiles2[index])
        'Generates one sample of data'
        # Select sample
        # Load data and get label


        drug1_id = self.df.iloc[index]['Drug1']
        drug2_id = self.df.iloc[index]['Drug2']

        d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input = drug_embedding(drug1_id)
        p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input = drug_embedding(drug2_id)
        label = self.labels[index]

        return d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,\
               p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,\
               label, (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2


def drug2emb_encoder(x):
    ## Sequence encoder parameter
    vocab_path = './ESPF/drug_codes_chembl.txt'
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    sub_csv = pd.read_csv('./ESPF/subword_units_map_chembl.csv')
    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
    max_d = 50
    t1 = dbpe.process_line(x).split()
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])
    except:
        i1 = np.array([0])
        print('error:', x)

    l = len(i1)

    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))

    else:
        i = i1[:max_d]
        input_mask = [1] * max_d
    return i, np.asarray(input_mask)

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]
    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])
    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1
    # print(adjacency.shape,node_features.shape,edge_features.shape)
    # (9, 9)(9, 75)(9, 9, 4)
    # (16, 16)(16, 75)(16, 16, 4)

    return adjacency, node_features, edge_features

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)

# def molgraph_collate_fn(data):
#     n_samples = len(data)
#     (adj_1, node_fts_1, edge_fts_1),(adj_2, node_fts_2, edge_fts_2),targets_0,d1,d2 ,mask_1,mask_2= data[0]
#     n_nodes_largest_graph_1 = max(map(lambda sample: sample[0][0].shape[0], data))
#     n_nodes_largest_graph_2 = max(map(lambda sample: sample[1][0].shape[0], data))
#
#     n_node_fts_1 = node_fts_1.shape[1]
#     n_edge_fts_1 = edge_fts_1.shape[2]
#     n_node_fts_2 = node_fts_2.shape[1]
#     n_edge_fts_2 = edge_fts_2.shape[2]
#     n_targets = 1
#     n_emb= d1.shape[0]
#     n_mask=mask_1.shape[0]
#
#     adjacency_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1)
#     node_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_node_fts_1)
#     edge_tensor_1 = torch.zeros(n_samples, n_nodes_largest_graph_1, n_nodes_largest_graph_1, n_edge_fts_1)
#
#     adjacency_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2)
#     node_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_node_fts_2)
#     edge_tensor_2 = torch.zeros(n_samples, n_nodes_largest_graph_2, n_nodes_largest_graph_2, n_edge_fts_2)
#
#     target_tensor = torch.zeros(n_samples, n_targets)
#     d1_emb_tensor=torch.zeros(n_samples, n_emb)
#     d2_emb_tensor = torch.zeros(n_samples, n_emb)
#     mask_1_tensor=torch.zeros(n_samples, n_mask)
#     mask_2_tensor = torch.zeros(n_samples, n_mask)
#
#
#     for i in range(n_samples):
#         (adj_1, node_fts_1, edge_fts_1),(adj_2, node_fts_2, edge_fts_2),target,d1,d2,mask_1,mask_2= data[i]
#         n_nodes_1 = adj_1.shape[0]
#         n_nodes_2 = adj_2.shape[0]
#
#
#         adjacency_tensor_1[i, :n_nodes_1, :n_nodes_1] = torch.Tensor(adj_1)
#         node_tensor_1[i, :n_nodes_1, :] = torch.Tensor(node_fts_1)
#         edge_tensor_1[i, :n_nodes_1, :n_nodes_1, :] = torch.Tensor(edge_fts_1)
#
#         adjacency_tensor_2[i, :n_nodes_2, :n_nodes_2] = torch.Tensor(adj_2)
#         node_tensor_2[i, :n_nodes_2, :] = torch.Tensor(node_fts_2)
#         edge_tensor_2[i, :n_nodes_2, :n_nodes_2, :] = torch.Tensor(edge_fts_2)
#
#         target_tensor[i]=torch.tensor(target)
#         d1_emb_tensor[i] = torch.IntTensor(d1)
#         d2_emb_tensor[i] = torch.IntTensor(d2)
#         mask_1_tensor[i] = torch.tensor(mask_1)
#         mask_2_tensor[i] = torch.tensor(mask_2)
#
#
#     return adjacency_tensor_1, node_tensor_1, edge_tensor_1, adjacency_tensor_2, node_tensor_2, edge_tensor_2,target_tensor,d1_emb_tensor,d2_emb_tensor,mask_1_tensor,mask_2_tensor
#

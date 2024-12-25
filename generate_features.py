import torch
from rdkit import Chem
from torch_geometric.data import Data

def get_node_features(mol):
    features = []
    for atom in mol.GetAtoms():
        features.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            int(atom.GetHybridization()),
            atom.GetIsAromatic()
        ])
    return torch.tensor(features, dtype=torch.float)

def get_edge_features(mol):
    edge_features = []
    edge_indices = []
    for bond in mol.GetBonds():
        edge_indices.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        edge_features.append([
            bond.GetBondTypeAsDouble(),
            bond.GetIsConjugated(),
            bond.IsInRing()
        ])
    return (
        torch.tensor(edge_indices, dtype=torch.long).t().contiguous(), 
        torch.tensor(edge_features, dtype=torch.float)
    )

def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    node_features = get_node_features(mol)
    edge_index, edge_features = get_edge_features(mol)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

if __name__ == "__main__":
    smiles = "CCO"
    graph = smiles_to_graph(smiles)
    print("Node Features:\n", graph.x)
    print("Edge Features:\n", graph.edge_attr)
    print("Edge Indices:\n", graph.edge_index)

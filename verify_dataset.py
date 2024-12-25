import torch
from rdkit import Chem
from rdkit.Chem import Draw

def load_processed_data(data_path="./data/processed/qm9_processed.pt"):
    return torch.load(data_path, weights_only=False)

def visualize_molecule(graph):
    mol = Chem.RWMol()
    atom_map = {}

    for i, atom_features in enumerate(graph.x):
        atomic_num = int(atom_features[0].item())
        atom_map[i] = mol.AddAtom(Chem.Atom(atomic_num))
    
    for edge_idx, (start, end) in enumerate(graph.edge_index.t().tolist()):
        bond_type = graph.edge_attr[edge_idx, 0].item()
        if bond_type == 1:
            rdkit_bond_type = Chem.rdchem.BondType.SINGLE
        elif bond_type == 2:
            rdkit_bond_type = Chem.rdchem.BondType.DOUBLE
        elif bond_type == 3:
            rdkit_bond_type = Chem.rdchem.BondType.TRIPLE
        elif bond_type == 4:
            rdkit_bond_type = Chem.rdchem.BondType.AROMATIC
        else:
            continue
        mol.AddBond(atom_map[start], atom_map[end], rdkit_bond_type)

    Chem.SanitizeMol(mol)

    return mol

if __name__ == "__main__":
    dataset = load_processed_data()
    print(f"Loaded {len(dataset)} molecular graphs.")

    first_graph = dataset[100]
    mol = visualize_molecule(first_graph)

    img = Draw.MolToImage(mol, size=(300, 300))
    img.show()

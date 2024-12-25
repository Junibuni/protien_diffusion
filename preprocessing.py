import os
import torch
from torch_geometric.datasets import QM9
from torch_geometric.data import DataLoader
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

DATA_DIR = "./data/qm9"

def load_qm9_dataset():
    dataset = QM9(root=DATA_DIR)
    return dataset

def save_processed_data(dataset, output_dir="./data/processed"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    torch.save(dataset, os.path.join(output_dir, "qm9_processed.pt"))

if __name__ == "__main__":
    dataset = load_qm9_dataset()
    print(f"Dataset loaded with {len(dataset)} molecules.")
    save_processed_data(dataset)

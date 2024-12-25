from torch_geometric.data import Data
from generate_features import smiles_to_graph
from preprocessing import load_qm9_dataset, save_processed_data

def preprocess_dataset(dataset):
    processed_data = []
    for data in dataset:
        smiles = data.smiles
        graph = smiles_to_graph(smiles)
        if graph is not None:
            processed_data.append(graph)
    return processed_data

if __name__ == "__main__":
    dataset = load_qm9_dataset()
    processed_data = preprocess_dataset(dataset)
    print(f"Processed {len(processed_data)} molecules.")
    save_processed_data(processed_data)

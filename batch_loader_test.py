from torch_geometric.data import DataLoader
from verify_dataset import load_processed_data

if __name__ == "__main__":
    dataset = load_processed_data()
    print(f"Loaded dataset with {len(dataset)} molecules.")

    batch_size = 32
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch_idx, batch in enumerate(data_loader):
        print(f"Batch {batch_idx + 1}:")
        print(f"Number of graphs: {batch.num_graphs}")
        print(f"Node feature shape: {batch.x.shape}")
        print(f"Edge index shape: {batch.edge_index.shape}")
        print(f"Edge attribute shape: {batch.edge_attr.shape}")
        break

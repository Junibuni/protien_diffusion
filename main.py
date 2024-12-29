import torch
from src.model.unet import GraphUNet
from src.train import train_model
from torch_geometric.loader import DataLoader

dataset = torch.load("./data/processed/qm9_processed.pt", weights_only=False)
data_loader = DataLoader(dataset, batch_size=128, shuffle=True)

model = GraphUNet(node_dim=4, edge_dim=3, hidden_dim=128, time_embed_dim=32, num_layers=6)
train_model(model, data_loader, num_epochs=50, device="cuda" if torch.cuda.is_available() else "cpu")
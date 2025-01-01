import os
import json
from tqdm import tqdm

import torch
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import QM9

from src.model.model import HybridSchNetMPNN

DATA_DIR = "./data/qm9"

model = HybridSchNetMPNN(
    node_dim=11, edge_dim=4, hidden_dim=128, output_dim=19,
    num_schnet_blocks=3, num_mpnn_layers=3
)

dataset = QM9(root=DATA_DIR)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

save_dir = "result"
os.makedirs(save_dir, exist_ok=True)

def train(model, loader, optimizer, criterion, epoch, total_epochs, loss_history):
    model.train()
    total_loss = 0.0
    batch_losses = []

    with tqdm(total=len(loader), desc=f"Epoch {epoch+1}/{total_epochs}", unit="batch") as pbar:
        for data in loader:
            optimizer.zero_grad()
            pred = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(pred, data.y)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{total_loss / (pbar.n + 1):.4f}")
            pbar.update(1)

    avg_loss = total_loss / len(loader)
    loss_history["epoch_losses"].append(avg_loss)
    loss_history["batch_losses"].append(batch_losses)

    return avg_loss

loss_history = {
    "epoch_losses": [],
    "batch_losses": []
}

total_epochs = 100
for epoch in range(total_epochs):
    avg_loss = train(model, data_loader, optimizer, criterion, epoch, total_epochs, loss_history)
    print(f"Epoch {epoch+1}/{total_epochs} completed. Average Loss: {avg_loss:.4f}")

model_save_path = os.path.join(save_dir, "final_model.pt")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

loss_history_save_path = os.path.join(save_dir, "loss_history.json")
with open(loss_history_save_path, "w") as f:
    json.dump(loss_history, f, indent=4)
print(f"Loss history saved to {loss_history_save_path}")

training_metadata = {
    "total_epochs": total_epochs,
    "batch_size": data_loader.batch_size,
    "optimizer": str(optimizer),
    "criterion": str(criterion),
    "final_avg_loss": loss_history["epoch_losses"][-1]
}
metadata_save_path = os.path.join(save_dir, "training_metadata.json")
with open(metadata_save_path, "w") as f:
    json.dump(training_metadata, f, indent=4)
print(f"Training metadata saved to {metadata_save_path}")
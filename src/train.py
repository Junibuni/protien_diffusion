from src.model.unet import GraphUNet
from torch_geometric.loader import DataLoader
import torch
import torch.optim as optim

def add_noise(x, t, noise_level, batch):
    noise = torch.randn_like(x)
    noise_scale = noise_level[t][batch].unsqueeze(-1)
    noisy_x = x + noise_scale * noise
    return noisy_x, noise

def train_model(model, data_loader, num_epochs, device):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    noise_level = torch.linspace(1e-4, 0.02, steps=1000).to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            batch = batch.to(device)
            t = torch.randint(0, 1000, (batch.num_graphs,), device=device)
            noisy_x, noise = add_noise(batch.x, t, noise_level, batch.batch)

            # Forward pass
            predicted_noise = model(noisy_x, batch.edge_index, batch.edge_attr, t, batch.batch)
            loss = ((predicted_noise - noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    dataset = torch.load("./data/processed/qm9_processed.pt")
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = GraphUNet(node_dim=4, edge_dim=3, hidden_dim=128, time_embed_dim=32, num_layers=3)
    train_model(model, data_loader, num_epochs=10, device="cuda" if torch.cuda.is_available() else "cpu")

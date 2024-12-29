import torch
import torch.nn as nn
from src.model.diffusion import TimeEmbedding, GraphEmbeddingLayer

class GraphUNet(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, time_embed_dim, num_layers=3):
        super(GraphUNet, self).__init__()
        self.num_layers = num_layers
        self.time_embedding = TimeEmbedding(time_embed_dim)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            GraphEmbeddingLayer(node_dim + time_embed_dim, edge_dim, hidden_dim)
        ])
        for _ in range(num_layers - 1):
            self.encoder_layers.append(GraphEmbeddingLayer(hidden_dim, edge_dim, hidden_dim))

        # Bottleneck
        self.bottleneck = GraphEmbeddingLayer(hidden_dim, edge_dim, hidden_dim)

        # Decoder
        self.decoder_layers = nn.ModuleList([
            GraphEmbeddingLayer(hidden_dim, edge_dim, hidden_dim, use_skip=True)
        ])
        for _ in range(num_layers - 1):
            self.decoder_layers.append(GraphEmbeddingLayer(2 * hidden_dim, edge_dim, hidden_dim))

        # Output
        self.output_layer = nn.Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, t, batch):
        t_emb = self.time_embedding(t)
        t_emb = t_emb[batch]

        x = torch.cat([x, t_emb], dim=-1)

        # Encoder
        skip_connections = []
        for layer in self.encoder_layers:
            x = layer(x, edge_index, edge_attr)
            skip_connections.append(x)

        # Bottleneck
        x = self.bottleneck(x, edge_index, edge_attr)

        # Decoder
        for layer, skip in zip(self.decoder_layers, reversed(skip_connections)):
            #print(f"[Decoder] x shape before concatenation: {x.shape}, skip shape: {skip.shape}")
            x = torch.cat([x, skip], dim=-1)
            #print(f"[Decoder] x shape after concatenation: {x.shape}")
            x = layer(x, edge_index, edge_attr)

        return self.output_layer(x)

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super(TimeEmbedding, self).__init__()
        self.embed_dim = embed_dim

    def forward(self, t):
        half_dim = self.embed_dim // 2
        device = t.device
        emb = torch.exp(-torch.arange(half_dim, dtype=torch.float32, device=device) * (10000 ** (-2 / half_dim)))
        emb = t[:, None] * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)

class GraphEmbeddingLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, out_dim, use_skip=False):
        super(GraphEmbeddingLayer, self).__init__(aggr="add")
        input_size = 2 * node_dim if not use_skip else 3 * node_dim
        self.edge_mlp = torch.nn.Linear(edge_dim, node_dim)
        self.node_mlp = torch.nn.Linear(input_size, out_dim)

    def forward(self, x, edge_index, edge_attr):
        #print(f"[Forward] x: {x.shape}, edge_index: {edge_index.shape}, edge_attr: {edge_attr.shape}")
        edge_attr = self.edge_mlp(edge_attr)
        #print(f"[Forward] edge_attr after edge_mlp: {edge_attr.shape}")
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        #print(f"[Message] x_j: {x_j.shape}, edge_attr: {edge_attr.shape}")
        message = torch.cat([x_j, edge_attr], dim=-1)
        #print(f"[Message] Concatenated message: {message.shape}")
        return message

    def update(self, aggr_out):
        #print(f"[Update] aggr_out: {aggr_out.shape}")
        updated = self.node_mlp(aggr_out)
        #print(f"[Update] Updated node features: {updated.shape}")
        return updated


class DiffusionModel(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, time_embed_dim):
        super(DiffusionModel, self).__init__()
        self.time_embedding = TimeEmbedding(time_embed_dim)
        self.node_gnn1 = GraphEmbeddingLayer(node_dim + time_embed_dim, edge_dim, hidden_dim)
        self.node_gnn2 = GraphEmbeddingLayer(hidden_dim, edge_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, node_dim)

    def forward(self, x, edge_index, edge_attr, t, batch):
        t_emb = self.time_embedding(t)
        t_emb = t_emb[batch]

        x = torch.cat([x, t_emb], dim=-1)

        x = self.node_gnn1(x, edge_index, edge_attr)
        x = self.node_gnn2(x, edge_index, edge_attr)

        return self.output_layer(x)

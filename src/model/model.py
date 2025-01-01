import torch
from torch.nn import Linear, ModuleList, SiLU, ReLU
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch_geometric.utils import add_self_loops

class SchNetInteractionBlock(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super(SchNetInteractionBlock, self).__init__(aggr='add')
        self.filter_network = torch.nn.Sequential(
            Linear(edge_dim, node_dim),
            SiLU(),
            Linear(node_dim, node_dim)
        )
        self.dense_network = Linear(node_dim, node_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_weight = self.filter_network(edge_attr)
        return self.propagate(edge_index, x=x, edge_weight=edge_weight)

    def message(self, x_i, x_j, edge_weight):
        return edge_weight * (x_i + x_j)

    def update(self, aggr_out):
        return self.dense_network(aggr_out)


class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim):
        super(MPNNLayer, self).__init__(aggr="add")
        self.edge_mlp = torch.nn.Linear(edge_dim, node_dim)
        self.node_mlp = torch.nn.Linear(2*node_dim, node_dim)

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


class HybridSchNetMPNN(torch.nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim, output_dim, num_schnet_blocks, num_mpnn_layers):
        super(HybridSchNetMPNN, self).__init__()

        self.node_embedding = Linear(node_dim, hidden_dim)
        self.edge_embedding = Linear(edge_dim, hidden_dim)

        self.schnet_blocks = ModuleList(
            [SchNetInteractionBlock(hidden_dim, hidden_dim) for _ in range(num_schnet_blocks)]
        )

        self.mpnn_layers = ModuleList(
            [MPNNLayer(hidden_dim, hidden_dim) for _ in range(num_mpnn_layers)]
        )

        self.global_pool = global_mean_pool
        self.fc1 = Linear(hidden_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, output_dim)
        self.activation = ReLU()

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.activation(self.node_embedding(x))
        edge_attr = self.activation(self.edge_embedding(edge_attr))

        for schnet_block in self.schnet_blocks:
            x = self.activation(schnet_block(x, edge_index, edge_attr))

        for mpnn_layer in self.mpnn_layers:
            x = self.activation(mpnn_layer(x, edge_index, edge_attr))

        x = self.global_pool(x, batch)

        x = self.activation(self.fc1(x))
        return self.fc2(x)

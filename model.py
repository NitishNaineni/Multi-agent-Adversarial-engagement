import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.transforms import VirtualNode
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils.convert import from_networkx

class GAT(torch.nn.Module):
    def __init__(self,num_gat_layers,gat_out_channels,num_agents,num_actions):
        super(GAT,self).__init__()

        self.num_gat_layers = num_gat_layers
        self.gat = GATv2Conv(in_channels=5,out_channels=gat_out_channels,edge_dim=1,dropout=0.5)
        self.propaGAT = GATv2Conv(in_channels=gat_out_channels,out_channels=gat_out_channels,edge_dim=1,dropout=0.5)
        self.fc1 = nn.Linear(gat_out_channels*(num_actions+num_agents),128)
        self.fc2 = nn.Linear(128,128)

        self.val1 = nn.Linear(128,128)
        self.adv1 = nn.Linear(128,128)
        self.val2 = nn.Linear(128,1)
        self.adv2 = nn.Linear(128,num_actions)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def forward(self,data,agent_node,actions):
        x = data.x.to(self.device)
        edge_index = data.edge_index.to(self.device)
        edge_weights = data.length.float().to(self.device)
        edge_weights = edge_weights / edge_weights.max()

        x = self.gat(x,edge_index,edge_attr=edge_weights)

        for i in range(self.num_gat_layers):
            x = self.propaGAT(x,edge_index,edge_attr=edge_weights)

        agent_node_encoding = x[agent_node]
        action_node_encoding = x[actions]
        node_embeddings = torch.cat([agent_node_encoding,action_node_encoding]).flatten()

        x = F.dropout(F.relu(self.fc1(node_embeddings)))
        x = F.dropout(F.relu(self.fc2(x)))

        v = F.dropout(F.relu(self.val1(x)))
        v = self.val2(v)

        a = F.dropout(F.relu(self.adv1(x)))
        a = self.adv2(a)

        return v + a - a.mean()
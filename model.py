import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ETSA(nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, num_gcn_layers=1, attention_type='sigmoid', dropout_rate=0.5):
        super(ETSA, self).__init__()

        self.embedding_layer = nn.Linear(num_features, hidden_dim)
        self.num_gcn_layers = num_gcn_layers
        self.gcn_layers = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_gcn_layers)])

        # Experiment with attention mechanisms
        if attention_type == 'sigmoid':
            self.teacher_attention = nn.Linear(hidden_dim, 1)
            self.student_attention = nn.Linear(hidden_dim, 1)
        elif attention_type == 'softmax':
            self.teacher_attention = nn.Linear(hidden_dim, hidden_dim)
            self.student_attention = nn.Linear(hidden_dim, hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=dropout_rate)

        self.final_gcn_layer = GCNConv(hidden_dim, num_classes)

    def forward(self, x, edge_index):
        # Embedding Layer
        x = self.embedding_layer(x)

        # GCN Layers with dropout
        for i in range(self.num_gcn_layers):
            x = F.relu(self.gcn_layers[i](x, edge_index))
            x = self.dropout(x)

        # Teacher Attention
        teacher_attention_weights = torch.sigmoid(self.teacher_attention(x))
        teacher_representation = torch.sum(x * teacher_attention_weights, dim=0)

        # Student Attention
        student_attention_weights = torch.sigmoid(self.student_attention(x))
        student_representation = torch.sum(x * student_attention_weights, dim=0)

        # Combine Teacher and Student Representations
        combined_representation = teacher_representation + student_representation

        # Final GCN Layer
        x = self.final_gcn_layer(x, edge_index)

        return x, combined_representation

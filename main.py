# main.py
import torch
import torch.nn as nn
from model import ETSA
from utils import setup_optimizer_and_scheduler
from train import train_model
from data_preprocessing import load_dataset, prepare_data

# Specify the dataset name ('cora', 'pubmed', or 'citeseer')
dataset_name = 'cora'

# Load dataset
graph_dataset = load_dataset(dataset_name)

# Set random seed for reproducibility
torch.manual_seed(42)

# Prepare data
data_loader = prepare_data(graph_dataset)

# Instantiate the ETSA model
num_features = graph_dataset.num_features
hidden_dim = 32
num_classes = graph_dataset.num_classes
num_gcn_layers = 2
attention_type = 'sigmoid'
dropout_rate = 0.5

# Instantiate the ETSA model
model = ETSA(num_features=num_features, hidden_dim=hidden_dim, num_classes=num_classes,
             num_gcn_layers=num_gcn_layers, attention_type=attention_type, dropout_rate=dropout_rate)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer, scheduler = setup_optimizer_and_scheduler(model, learning_rate=0.01, factor=0.1, patience=5)

# Early stopping
best_val_accuracy = 0.0
patience = 10

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    train_model(model, data_loader, criterion, optimizer, scheduler, best_val_accuracy, patience, epoch, num_epochs)





















































































































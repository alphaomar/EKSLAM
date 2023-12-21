# train.py
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader
from sklearn.metrics import accuracy_score

def train_model(model, data_loader, criterion, optimizer, scheduler, best_val_accuracy, patience, epoch, num_epochs):
    # Early stopping
    counter = 0

    model.train()
    optimizer.zero_grad()

    # Use DataLoader for mini-batch training
    for data in data_loader:
        out, _ = model(data.x, data.edge_index)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()

    # Evaluation on validation set
    model.eval()
    with torch.no_grad():
        for data in data_loader:
            val_logits, _ = model(data.x, data.edge_index)
            val_pred_labels = val_logits.argmax(dim=1)
            val_accuracy = accuracy_score(data.y, val_pred_labels)

    print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, Validation Accuracy: {val_accuracy:.4f}')

    # Learning rate scheduler step
    scheduler.step(val_accuracy)

    # Early stopping
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping. No improvement in validation accuracy.")
            return


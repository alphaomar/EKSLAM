# utils.py
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def setup_optimizer_and_scheduler(model, learning_rate=0.01, factor=0.1, patience=5):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=factor, patience=patience, verbose=True)
    return optimizer, scheduler

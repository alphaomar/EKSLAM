# data_processing.py
import torch
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.data import DataLoader


def load_dataset(dataset_name):
    if dataset_name == 'cora':
        return Planetoid(root='data/CORA', name='CORA')
    elif dataset_name == 'pubmed':
        return Planetoid(root='data/PubMed', name='PubMed')
    elif dataset_name == 'citeseer':
        return CitationFull(root='data/CiteSeer', name='CiteSeer')
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")


def prepare_data(dataset):
    # DataLoader returns an iterable, not subscriptable
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return data_loader

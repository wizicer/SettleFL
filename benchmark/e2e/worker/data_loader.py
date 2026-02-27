import os
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, ConcatDataset


def load_data(dataset_name, data_dir, participant_count, max_data_size_per_participant):
    if dataset_name == 'mnist':
        return load_mnist(data_dir, participant_count, max_data_size_per_participant)
    elif dataset_name == 'cifar10':
        return load_cifar10(data_dir, participant_count, max_data_size_per_participant)
    elif dataset_name == 'fashion-mnist':
        return load_fashion_mnist(data_dir, participant_count, max_data_size_per_participant)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

def get_transforms(dataset_name):
    if dataset_name in ['mnist', 'fashion-mnist']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_name == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def split_data_for_participants(dataset, num_participants, max_samples_per_participant):
    num_samples = len(dataset)
    samples_per_participant = min(max_samples_per_participant, num_samples // num_participants)
    
    if samples_per_participant == 0:
        raise ValueError(f"Not enough data for {num_participants} participants.")

    # Shuffle indices
    indices = np.random.permutation(num_samples)
    
    participant_datasets = []
    for i in range(num_participants):
        start_idx = i * samples_per_participant
        end_idx = start_idx + samples_per_participant
        participant_indices = indices[start_idx:end_idx]
        participant_datasets.append(Subset(dataset, participant_indices))
        
    return participant_datasets

def load_mnist(data_dir, participant_count, max_data_size):
    transform = get_transforms('mnist')
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    
    participant_datasets = split_data_for_participants(train_dataset, participant_count, max_data_size)
    return participant_datasets, test_dataset

def load_cifar10(data_dir, participant_count, max_data_size):
    transform = get_transforms('cifar10')
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    participant_datasets = split_data_for_participants(train_dataset, participant_count, max_data_size)
    return participant_datasets, test_dataset

def load_fashion_mnist(data_dir, participant_count, max_data_size):
    transform = get_transforms('fashion-mnist')
    train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

    participant_datasets = split_data_for_participants(train_dataset, participant_count, max_data_size)
    return participant_datasets, test_dataset

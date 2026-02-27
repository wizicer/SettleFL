import argparse
import json
import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models import get_model
from data_loader import load_data


def train(model, train_loader, epochs, learning_rate, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()

    # Return metrics from the last batch of the last epoch
    accuracy = (output.argmax(dim=1) == target).float().mean().item()
    return {'loss': loss.item(), 'accuracy': accuracy}

def evaluate(model, test_loader, device):
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    print(f"Test Loss: {test_loss}, correct: {correct}, total: {len(test_loader.dataset)}")
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return {'testLoss': test_loss, 'testAccuracy': accuracy}

def aggregate_weights(weight_files):
    aggregated_weights = None
    num_participants = len(weight_files)

    for file_path in weight_files:
        weights = torch.load(file_path)
        if aggregated_weights is None:
            aggregated_weights = {name: torch.zeros_like(param) for name, param in weights.items()}
        
        for name, param in weights.items():
            aggregated_weights[name] += param

    for name in aggregated_weights:
        # Only average floating-point tensors. Integer tensors (like batch norm tracking) should not be averaged.
        if aggregated_weights[name].is_floating_point():
            aggregated_weights[name] /= num_participants

    return aggregated_weights

def main():
    parser = argparse.ArgumentParser(description='PyTorch Worker for Federated Learning')
    parser.add_argument('action', choices=['train', 'evaluate', 'aggregate'])
    parser.add_argument('--model', required=True, help='Model architecture')
    parser.add_argument('--dataset', help='Dataset name')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--participant-id', type=int, help='ID of the current participant')
    parser.add_argument('--participant-count', type=int, help='Total number of participants')
    parser.add_argument('--max-data-size', type=int, help='Max data size per participant')
    parser.add_argument('--weights-in', help='Path to input weights file (for train/evaluate)')
    parser.add_argument('--weights-out', help='Path to output weights file (for train/aggregate)')
    parser.add_argument('--results-out', help='Path to output results JSON file')
    parser.add_argument('--participant-weights-in', nargs='*', help='Paths to participant weight files for aggregation')
    parser.add_argument('--gpu-id', type=int, default=0, help='GPU device ID to use')
    parser.add_argument('--data-dir', default=os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data'), help='Path to the dataset directory')

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    model = get_model(args.model)
    if args.action == 'train':
        participant_datasets, _ = load_data(args.dataset, args.data_dir, args.participant_count, args.max_data_size)
        train_loader = DataLoader(participant_datasets[args.participant_id], batch_size=32, shuffle=True)
        
        if args.weights_in and os.path.exists(args.weights_in) and os.path.getsize(args.weights_in) > 0:
            model.load_state_dict(torch.load(args.weights_in, map_location=device))
        
        metrics = train(model, train_loader, args.epochs, args.learning_rate, device)
        
        weights_out_path = args.weights_out
        # Move to CPU before saving to ensure device-independent weights
        model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(model_state, weights_out_path)
        with open(args.results_out, 'w') as f:
            json.dump(metrics, f)

    elif args.action == 'evaluate':
        _, test_dataset = load_data(args.dataset, args.data_dir, 1, 1) # Counts don't matter here
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        model.load_state_dict(torch.load(args.weights_in, map_location=device))
        metrics = evaluate(model, test_loader, device)
        
        with open(args.results_out, 'w') as f:
            json.dump(metrics, f)

    elif args.action == 'aggregate':
        aggregated_weights = aggregate_weights(args.participant_weights_in)
        torch.save(aggregated_weights, args.weights_out)

if __name__ == '__main__':
    main()

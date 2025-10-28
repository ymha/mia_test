"""
Train shadow models for CNN classifier

These shadow models mimic the CNN architecture to generate
training data for the membership inference attack.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
import os
from cnn_classifier_model import CNNClassifier


def train_single_cnn_shadow(shadow_id, num_samples=10000,
                            epochs=100, batch_size=128, learning_rate=0.001,
                            weight_decay=1e-7,
                            device='cpu', save_dir='cnn_shadow_models',
                            target_indices=None):
    """
    Train a single shadow model with CNN architecture

    Args:
        shadow_id: ID of the shadow model
        num_samples: Number of training samples
        epochs: Number of epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to use
        save_dir: Directory to save shadow models
        target_indices: Indices used by target (to exclude)

    Returns:
        trained shadow model, training indices,
        final_train_accuracy, final_train_loss,
        final_test_accuracy, final_test_loss
    """
    torch.manual_seed(shadow_id)
    np.random.seed(shadow_id)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    # Get available indices (excluding target classifier's training data)
    all_indices = np.arange(len(train_dataset))
    if target_indices is not None:
        available_indices = np.setdiff1d(all_indices, target_indices)
    else:
        available_indices = all_indices

    # Randomly select num_samples from available indices
    if len(available_indices) < num_samples:
        raise ValueError(f"Not enough samples. Need {num_samples}, have {len(available_indices)}")

    indices = np.random.choice(available_indices, num_samples, replace=False)

    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    shadow_model = CNNClassifier().to(device)
    optimizer = optim.Adam(shadow_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    shadow_model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            logits = shadow_model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            shadow_model.eval()
            test_loss_e = 0
            test_correct_e = 0
            test_total_e = 0
            train_correct_e = 0
            train_total_e = 0

            with torch.no_grad():
                for data, labels in train_loader:
                    data = data.to(device)
                    labels = labels.long().to(device)
                    logits = shadow_model(data)
                    predicted = torch.argmax(logits, dim=1)
                    train_total_e += labels.size(0)
                    train_correct_e += (predicted == labels).sum().item()

            with torch.no_grad():
                for data, labels in test_loader:
                    data = data.to(device)
                    labels = labels.long().to(device)
                    logits = shadow_model(data)
                    loss = criterion(logits, labels)
                    test_loss_e += loss.item()
                    predicted = torch.argmax(logits, dim=1)
                    test_total_e += labels.size(0)
                    test_correct_e += (predicted == labels).sum().item()

            train_acc_e = 100 * train_correct_e / train_total_e
            test_acc_e = 100 * test_correct_e / test_total_e

            print(f"  [Shadow {shadow_id} Epoch {epoch+1}/{epochs}] Train Acc: {train_acc_e:.2f}% | Test Acc: {test_acc_e:.2f}%")
            shadow_model.train()

    shadow_model.eval()
    final_train_correct = 0
    final_train_total = 0
    final_train_loss = 0
    with torch.no_grad():
        for data, labels in train_loader:
            data = data.to(device)
            labels = labels.long().to(device)
            logits = shadow_model(data)
            loss = criterion(logits, labels)

            final_train_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            final_train_total += labels.size(0)
            final_train_correct += (predicted == labels).sum().item()

    final_train_accuracy = 100 * final_train_correct / final_train_total
    final_train_loss = final_train_loss / len(train_loader)

    final_test_correct = 0
    final_test_total = 0
    final_test_loss = 0
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.long().to(device)
            logits = shadow_model(data)
            loss = criterion(logits, labels)

            final_test_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            final_test_total += labels.size(0)
            final_test_correct += (predicted == labels).sum().item()

    final_test_accuracy = 100 * final_test_correct / final_test_total
    final_test_loss = final_test_loss / len(test_loader)

    print(f"  CNN shadow {shadow_id} final: Train Acc={final_train_accuracy:.2f}%, Test Acc={final_test_accuracy:.2f}% (Gap: {final_train_accuracy - final_test_accuracy:.2f}%)")

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cnn_shadow_{shadow_id}.pth')

    torch.save({
        'model_state_dict': shadow_model.state_dict(),
        'training_indices': indices,
        'shadow_id': shadow_id,
        'epochs': epochs,
        'final_train_accuracy': final_train_accuracy,
        'final_train_loss': final_train_loss,
        'final_test_accuracy': final_test_accuracy,
        'final_test_loss': final_test_loss
    }, save_path)

    return shadow_model, indices, final_train_accuracy, final_train_loss, final_test_accuracy, final_test_loss


def train_cnn_shadow_models(num_shadows=50, cnn_classifier_path='cnn_classifier.pth',
                           num_samples=10000, epochs=100,
                           batch_size=128, learning_rate=0.001, weight_decay=1e-7,
                           device='cpu', save_dir='cnn_shadow_models'):
    """
    Train multiple CNN shadow models for membership inference attack
    """
    print(f"Training {num_shadows} CNN shadow models")
    print(f"Each shadow model: 2 Conv+Pool layers, FC(128), Output(10)")
    print(f"Device: {device}")
    print()

    checkpoint = torch.load(cnn_classifier_path, map_location=device, weights_only=False)
    target_training_indices = checkpoint['training_indices']
    target_train_acc = checkpoint.get('final_train_accuracy', 'N/A')
    target_test_acc = checkpoint.get('final_test_accuracy', 'N/A')
    print(f"Target classifier uses {len(target_training_indices)} training samples")
    print(f"Target (for reference): Train Acc={target_train_acc}%, Test Acc={target_test_acc}%")
    print(f"Shadow models will be trained on disjoint data")
    print()

    all_indices = []
    all_train_accuracies = []
    all_train_losses = []
    all_test_accuracies = []
    all_test_losses = []

    for i in range(num_shadows):
        print(f"Training CNN shadow model {i+1}/{num_shadows}...")

        _, indices, train_acc, train_loss, test_acc, test_loss = train_single_cnn_shadow(
            shadow_id=i,
            num_samples=num_samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=device,
            save_dir=save_dir,
            target_indices=target_training_indices
        )
        all_indices.append(indices)
        all_train_accuracies.append(train_acc)
        all_train_losses.append(train_loss)
        all_test_accuracies.append(test_acc)
        all_test_losses.append(test_loss)
        print(f"✓ Shadow model {i+1} complete")

    np.save(os.path.join(save_dir, 'all_cnn_shadow_indices.npy'), np.array(all_indices))

    import json
    performance_summary = {
        'shadow_performances': [
            {
                'shadow_id': i,
                'train_accuracy': float(all_train_accuracies[i]),
                'train_loss': float(all_train_losses[i]),
                'test_accuracy': float(all_test_accuracies[i]),
                'test_loss': float(all_test_losses[i])
            }
            for i in range(num_shadows)
        ],
        'statistics': {
            'mean_train_accuracy': float(np.mean(all_train_accuracies)),
            'std_train_accuracy': float(np.std(all_train_accuracies)),
            'min_train_accuracy': float(np.min(all_train_accuracies)),
            'max_train_accuracy': float(np.max(all_train_accuracies)),
            'mean_train_loss': float(np.mean(all_train_losses)),
            'std_train_loss': float(np.std(all_train_losses)),

            'mean_test_accuracy': float(np.mean(all_test_accuracies)),
            'std_test_accuracy': float(np.std(all_test_accuracies)),
            'min_test_accuracy': float(np.min(all_test_accuracies)),
            'max_test_accuracy': float(np.max(all_test_accuracies)),
            'mean_test_loss': float(np.mean(all_test_losses)),
            'std_test_loss': float(np.std(all_test_losses))
        }
    }

    with open(os.path.join(save_dir, 'cnn_shadow_performance_summary.json'), 'w') as f:
        json.dump(performance_summary, f, indent=2)

    print(f"\n{'='*70}")
    print(f"CNN SHADOW MODELS PERFORMANCE SUMMARY")
    print(f"{'='*70}")
    print(f"Total shadow models trained: {num_shadows}")

    print(f"\nPerformance Statistics (Train):")
    print(f"  Mean Train Accuracy: {np.mean(all_train_accuracies):.2f}% (± {np.std(all_train_accuracies):.2f}%)")
    print(f"  Mean Train Loss:     {np.mean(all_train_losses):.4f} (± {np.std(all_train_losses):.4f})")

    print(f"\nPerformance Statistics (Test):")
    print(f"  Mean Test Accuracy:  {np.mean(all_test_accuracies):.2f}% (± {np.std(all_test_accuracies):.2f}%)")
    print(f"  Mean Test Loss:      {np.mean(all_test_losses):.4f} (± {np.std(all_test_losses):.4f})")

    mean_gap = np.mean(all_train_accuracies) - np.mean(all_test_accuracies)
    print(f"\nMean Overfitting Gap (Train Acc - Test Acc): {mean_gap:.2f}%")

    print(f"\nFiles saved to {save_dir}/:")
    print(f"  - cnn_shadow_{{id}}.pth: Shadow models")
    print(f"  - cnn_shadow_performance_summary.json: Performance metrics")
    print(f"{'='*70}")

    return all_indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train CNN shadow models')
    parser.add_argument('--num_shadows', type=int, default=50, help='Number of shadow models')
    parser.add_argument('--num_samples', type=int, default=10000, help='Samples per shadow')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-7, help='Weight decay')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')

    args = parser.parse_args()

    train_cnn_shadow_models(
        num_shadows=args.num_shadows,
        num_samples=args.num_samples,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device
    )

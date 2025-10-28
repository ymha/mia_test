"""
Train MLP classifier

This creates a classifier that is intentionally vulnerable to demonstrate
that membership inference attacks CAN work without privacy protections.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from mlp_classifier_model import MLPClassifier
import os


def train_mlp_classifier(num_samples=10000, hidden_dim=512, epochs=100,
                         batch_size=128, learning_rate=1e-3,
                         device='cuda' if torch.cuda.is_available() else 'cpu',
                         save_path='mlp_classifier.pth'):
    """
    Train MLP classifier

    This classifier is vulnerable to membership inference because:
    - No information bottleneck (trains on raw 3072-D pixels)
    - Large capacity (hidden_dim-256 architecture)
    - Long training (100 epochs) allows memorization

    Args:
        num_samples: Number of training samples
        hidden_dim: Hidden layer dimension
        epochs: Number of training epochs (more = more memorization)
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device to use
        save_path: Path to save model

    Returns:
        Trained classifier model and training indices
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    # Randomly select num_samples from the training set
    indices = np.random.choice(len(train_dataset), num_samples, replace=False)

    # Create training subset
    train_subset = Subset(train_dataset, indices)
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = MLPClassifier(hidden_dim=hidden_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    model.train()
    print(f"Training MLP classifier on {num_samples} CIFAR10 samples...")
    print(f"Hidden dim: {hidden_dim}, Epochs: {epochs}")
    print(f"Device: {device}")
    print("\nThis model is INTENTIONALLY VULNERABLE to membership inference!")
    print("Monitoring Train vs Test performance to check for overfitting gap...")
    print("="*70)

    final_train_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            data = data.view(-1, 3072).to(device)
            labels = labels.long().to(device)

            optimizer.zero_grad()
            logits = model(data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        avg_loss = train_loss / len(train_loader)
        accuracy = 100 * correct / total
        final_train_acc = accuracy

        # Evaluate on test set
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for data, labels in test_loader:
                data = data.view(-1, 3072).to(device)
                labels = labels.long().to(device)
                
                logits = model(data)
                loss = criterion(logits, labels)
                
                test_loss += loss.item()
                predicted = torch.argmax(logits, dim=1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_avg_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        final_test_acc = test_accuracy

        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_loss:.4f}, Train Acc: {accuracy:.2f}% | Test Loss: {test_avg_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    # Save the trained model
    save_dir = os.path.dirname(save_path)
    if save_dir:  # Only create directory if path includes a directory
        os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'hidden_dim': hidden_dim,
        'training_indices': indices,
        'epochs': epochs,
        'final_train_accuracy': final_train_acc,
        'final_test_accuracy': final_test_acc
    }, save_path)

    print(f"\n{'='*70}")
    print(f"MLP classifier training complete!")
    print(f"Model saved to {save_path}")
    print(f"Training indices: {len(indices)} samples")
    print(f"Final Train Accuracy: {final_train_acc:.2f}%")
    print(f"Final Test Accuracy:  {final_test_acc:.2f}%")

    overfitting_gap = final_train_acc - final_test_acc
    if overfitting_gap > 5.0:
        print(f"Overfitting detected! (Gap: {overfitting_gap:.2f}%) -> MIA should be effective.")
    else:
        print(f"Low overfitting (Gap: {overfitting_gap:.2f}%) -> MIA might be less effective.")

    print(f"{'='*70}")

    return model, indices


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train MLP classifier')
    parser.add_argument('--num_samples', type=int, default=10000, help='Number of training samples')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden layer dimension')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')

    args = parser.parse_args()

    model, indices = train_mlp_classifier(
        num_samples=args.num_samples,
        hidden_dim=args.hidden_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device
    )

    print(f"\n✓ Training completed with {len(indices)} samples")
    print(f"✓ Ready for membership inference attack testing")
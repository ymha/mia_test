import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import numpy as np
import os


class AttackModel(nn.Module):
    """
    Binary classifier to predict membership (in training set or not)
    """
    def __init__(self, input_dim):
        super(AttackModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x).squeeze(-1)
        return x

def train_attack_models(class_datasets, epochs=100, batch_size=256, learning_rate=1e-3,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        save_dir='attack_models'):
    """
    Train CLASS-SPECIFIC membership inference attack models

    Args:
        class_datasets: dict mapping class_id -> (X_train, y_train)
            - class 0: X_train shape (N_0, 11), y_train shape (N_0,)
            - class 1: X_train shape (N_1, 11), y_train shape (N_1,)
            - ...
            - class 9: X_train shape (N_9, 11), y_train shape (N_9,)
            Features: [class_label, prob_0, prob_1, ..., prob_9]
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning_rate
        device: Device to use
        save_dir: Directory to save attack models

    Returns:
        dict mapping class_id -> trained attack model
    """
    os.makedirs(save_dir, exist_ok=True)

    attack_models = {}

    for class_id in range(10):
        X_train, y_train = class_datasets[class_id]

        print(f"\n{'='*60}")
        print(f"Training attack model for Class {class_id}...")
        print(f"{'='*60}")

        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.astype(float))

        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Initialize attack model
        input_dim = X_train.shape[1]

        attack_model = AttackModel(input_dim).to(device)
        optimizer = optim.Adam(attack_model.parameters(), lr=learning_rate)
        criterion = nn.BCEWithLogitsLoss()

        # Training loop
        attack_model.train()
        for epoch in range(epochs):
            train_loss = 0
            correct = 0
            total = 0

            for features, labels in train_loader:
                features = features.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = attack_model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                predicted = (outputs > 0).long()
                total += labels.size(0)
                correct += (predicted == labels.long()).sum().item()

            avg_loss = train_loss / len(train_loader)
            accuracy = 100 * correct / total

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

        # Save attack model for this class
        save_path = os.path.join(save_dir, f'attack_model_class_{class_id}.pth')
        torch.save({
            'model_state_dict': attack_model.state_dict(),
            'input_dim': input_dim,
            'class_id': class_id
        }, save_path)

        print(f"Attack model for class {class_id} training complete! Saved to {save_path}")

        attack_models[class_id] = attack_model

    return attack_models


def evaluate_attack(attack_models, target_classifier, classifier_path='classifier_model.pth', latent_dim=20,
                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate CLASS-SPECIFIC membership inference attacks on the target classifier

    Args:
        attack_models: dict mapping class_id -> attack model
        target_classifier: Target classifier model
        classifier_path: Path to classifier model
        latent_dim: Latent dimension
        device: Device to use

    Returns:
        Dictionary of results
    """
    print("\n" + "="*60)
    print("EVALUATING MEMBERSHIP INFERENCE ATTACK")
    print("="*60)

    # Load target classifier checkpoint
    classifier_checkpoint = torch.load(classifier_path, map_location=device, weights_only=False)
    target_training_indices = classifier_checkpoint['training_indices']

    # Data preprocessing
    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST training set (for members)
    mnist_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    # Load MNIST test set (for non-members)
    mnist_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Extract features for members (from target's training data)
    member_dataset = torch.utils.data.Subset(mnist_train_dataset, target_training_indices)
    member_loader = DataLoader(member_dataset, batch_size=128, shuffle=False)
    member_features_by_class, _ = extract_features_from_model(target_classifier, member_loader, device)

    # Extract features for non-members (from MNIST test set)
    non_member_loader = DataLoader(mnist_test_dataset, batch_size=128, shuffle=False)
    non_member_features_by_class, _ = extract_features_from_model(target_classifier, non_member_loader, device)

    # Verify test data composition
    print(f"\nTest dataset composition:")
    total_members = sum(len(member_features_by_class[c]) for c in range(10))
    total_non_members = sum(len(non_member_features_by_class[c]) for c in range(10))
    print(f"  Total samples: {total_members + total_non_members}")
    print(f"  Members (IN): {total_members}")
    print(f"  Non-members (OUT): {total_non_members}")

    for class_id in range(10):
        class_members = len(member_features_by_class[class_id])
        class_non_members = len(non_member_features_by_class[class_id])
        print(f"  Class {class_id}: {class_members} IN + {class_non_members} OUT = {class_members + class_non_members} total")

    # Make predictions using CLASS-SPECIFIC attack models
    all_predictions = []
    all_probabilities = []
    all_true_labels = []
    all_class_labels = []

    for class_id in range(10):
        attack_model = attack_models[class_id]
        attack_model.eval()

        # Combine member and non-member features for this class
        X_test_class = np.vstack([
            member_features_by_class[class_id],
            non_member_features_by_class[class_id]
        ])
        y_test_class = np.concatenate([
            np.ones(len(member_features_by_class[class_id])),
            np.zeros(len(non_member_features_by_class[class_id]))
        ])

        X_test_tensor = torch.FloatTensor(X_test_class).to(device)

        with torch.no_grad():
            outputs = attack_model(X_test_tensor)
            probabilities_class = torch.sigmoid(outputs)
            predictions_class = (outputs > 0).long()

        predictions_class = predictions_class.cpu().numpy()
        probabilities_class = probabilities_class.cpu().numpy()

        all_predictions.append(predictions_class)
        all_probabilities.append(probabilities_class)
        all_true_labels.append(y_test_class)
        all_class_labels.extend([class_id] * len(y_test_class))

    # Combine all predictions and labels
    predictions = np.concatenate(all_predictions)
    predicted_probs = np.concatenate(all_probabilities, axis=0)
    y_test = np.concatenate(all_true_labels)

    # Calculate overall metrics
    correct = (predictions == y_test).sum()
    accuracy = 100 * correct / len(y_test)

    # True positives, false positives, true negatives, false negatives
    tp = ((predictions == 1) & (y_test == 1)).sum()
    fp = ((predictions == 1) & (y_test == 0)).sum()
    tn = ((predictions == 0) & (y_test == 0)).sum()
    fn = ((predictions == 0) & (y_test == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Calculate per-class metrics
    class_results = {}
    all_class_labels_np = np.array(all_class_labels)

    for class_id in range(10):
        class_mask = all_class_labels_np == class_id
        if np.sum(class_mask) == 0:
            continue

        class_pred = predictions[class_mask]
        class_true = y_test[class_mask]

        class_correct = (class_pred == class_true).sum()
        class_accuracy = 100 * class_correct / len(class_true)

        class_tp = ((class_pred == 1) & (class_true == 1)).sum()
        class_fp = ((class_pred == 1) & (class_true == 0)).sum()
        class_tn = ((class_pred == 0) & (class_true == 0)).sum()
        class_fn = ((class_pred == 0) & (class_true == 1)).sum()

        class_precision = class_tp / (class_tp + class_fp) if (class_tp + class_fp) > 0 else 0
        class_recall = class_tp / (class_tp + class_fn) if (class_tp + class_fn) > 0 else 0
        class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0

        class_results[f"Class {class_id}"] = {
            'accuracy': class_accuracy,
            'precision': class_precision,
            'recall': class_recall,
            'f1_score': class_f1,
            'samples': len(class_true)
        }

    print("\n" + "="*60)
    print("MEMBERSHIP INFERENCE ATTACK RESULTS")
    print("="*60)
    print(f"\nOverall Attack Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  True Negatives: {tn}")
    print(f"  False Negatives: {fn}")

    print(f"\n{'-'*60}")
    print("Per-Class Results:")
    print(f"{'-'*60}")
    for class_label, results in class_results.items():
        print(f"\nClass: {class_label} ({results['samples']} samples)")
        print(f"  Accuracy: {results['accuracy']:.2f}%")
        print(f"  Precision: {results['precision']:.4f}")
        print(f"  Recall: {results['recall']:.4f}")
        print(f"  F1 Score: {results['f1_score']:.4f}")

    print("="*60)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': {'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)},
        'predictions': predictions,
        'probabilities': predicted_probs,
        'true_labels': y_test,
        'class_results': class_results
    }


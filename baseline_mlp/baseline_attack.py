"""
Membership inference attack on baseline classifier

This script demonstrates that attacks WORK on classifiers without privacy protections.
"""

import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from baseline_classifier_model import BaselineClassifier
from membership_inference_attack import AttackModel, train_attack_models


def extract_baseline_features(model, data_loader, device='cpu'):
    """
    Extract features from baseline classifier for membership inference

    Features (11-D):
    - Class label (0-9)
    - Softmax probability for class 0
    - Softmax probability for class 1
    - ...
    - Softmax probability for class 9

    Returns:
        features_by_class: dict mapping class_id -> features array
        all_labels: array of class labels
    """
    model.eval()

    class_features = {i: [] for i in range(10)}
    all_labels = []

    with torch.no_grad():
        for data, labels in data_loader:
            data = data.view(-1, 3072).to(device)
            labels = labels.to(device)

            logits = model(data)
            probs = torch.softmax(logits, dim=1)

            probs_np = probs.cpu().numpy()
            labels_np = labels.cpu().numpy()

            for i in range(len(labels_np)):
                class_label = labels_np[i]
                probs_sample = probs_np[i]

                # Feature vector: [class_label, prob_0, prob_1, ..., prob_9]
                features = np.concatenate([[class_label], probs_sample])

                class_features[class_label].append(features)
                all_labels.append(class_label)

    features_by_class = {}
    for class_id in range(10):
        if len(class_features[class_id]) > 0:
            features_by_class[class_id] = np.array(class_features[class_id])
        else:
            features_by_class[class_id] = np.array([]).reshape(0, 11)

    return features_by_class, np.array(all_labels)


def prepare_baseline_attack_dataset(shadow_dir='baseline_shadow_models',
                                    baseline_classifier_path='baseline_classifier.pth',
                                    hidden_dim=512,
                                    device='cpu'):
    """
    Prepare attack dataset from baseline shadow models
    """
    print("Preparing attack dataset from baseline shadow models...")
    print("Features: 11-D (class_label, prob_0, prob_1, ..., prob_9)")
    print("Labels: 1=member (IN), 0=non-member (OUT)")
    print()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)

    checkpoint = torch.load(baseline_classifier_path, map_location=device, weights_only=False)
    target_training_indices = set(checkpoint['training_indices'])
    print(f"Target model trained on {len(target_training_indices)} samples")
    print()

    class_attack_features = {i: [] for i in range(10)}
    class_attack_labels = {i: [] for i in range(10)}

    shadow_files = sorted([f for f in os.listdir(shadow_dir) if f.startswith('baseline_shadow_') and f.endswith('.pth')])

    for shadow_file in shadow_files:
        shadow_path = os.path.join(shadow_dir, shadow_file)
        shadow_checkpoint = torch.load(shadow_path, map_location=device, weights_only=False)
        shadow_id = shadow_checkpoint['shadow_id']
        shadow_indices = shadow_checkpoint['training_indices']

        shadow_model = BaselineClassifier(hidden_dim=hidden_dim).to(device)
        shadow_model.load_state_dict(shadow_checkpoint['model_state_dict'])
        shadow_model.eval()

        all_indices = np.arange(len(cifar10_dataset))
        non_member_indices = np.setdiff1d(all_indices, shadow_indices)
        non_member_indices = np.setdiff1d(non_member_indices, list(target_training_indices))

        non_member_indices = np.random.choice(non_member_indices,
                                             min(len(shadow_indices), len(non_member_indices)),
                                             replace=False)

        member_dataset = torch.utils.data.Subset(cifar10_dataset, shadow_indices)
        member_loader = DataLoader(member_dataset, batch_size=128, shuffle=False)
        member_features_by_class, _ = extract_baseline_features(shadow_model, member_loader, device)

        non_member_dataset = torch.utils.data.Subset(cifar10_dataset, non_member_indices)
        non_member_loader = DataLoader(non_member_dataset, batch_size=128, shuffle=False)
        non_member_features_by_class, _ = extract_baseline_features(shadow_model, non_member_loader, device)

        for class_id in range(10):
            if len(member_features_by_class[class_id]) > 0:
                class_attack_features[class_id].append(member_features_by_class[class_id])
                class_attack_labels[class_id].append(np.ones(len(member_features_by_class[class_id])))

            if len(non_member_features_by_class[class_id]) > 0:
                class_attack_features[class_id].append(non_member_features_by_class[class_id])
                class_attack_labels[class_id].append(np.zeros(len(non_member_features_by_class[class_id])))

        if (shadow_id + 1) % 10 == 0:
            print(f"Processed {shadow_id + 1} shadow models...")

    print(f"\nBalancing attack datasets...")
    class_datasets = {}

    for class_id in range(10):
        X_class = np.vstack(class_attack_features[class_id])
        y_class = np.concatenate(class_attack_labels[class_id])

        n_members = np.sum(y_class == 1)
        n_non_members = np.sum(y_class == 0)
        n_balanced = min(n_members, n_non_members)

        print(f"  Class {class_id}: {n_members} members, {n_non_members} non-members")
        print(f"    -> Sampling {n_balanced} from each")

        member_indices = np.where(y_class == 1)[0]
        non_member_indices = np.where(y_class == 0)[0]

        np.random.seed(42 + class_id)
        balanced_member_indices = np.random.choice(member_indices, n_balanced, replace=False)
        balanced_non_member_indices = np.random.choice(non_member_indices, n_balanced, replace=False)

        balanced_indices = np.concatenate([balanced_member_indices, balanced_non_member_indices])
        np.random.shuffle(balanced_indices)

        X_train_class = X_class[balanced_indices]
        y_train_class = y_class[balanced_indices]

        class_datasets[class_id] = (X_train_class, y_train_class)

    print(f"\nBalanced attack dataset prepared:")
    for class_id in range(10):
        print(f"  Class {class_id}: {len(class_datasets[class_id][0])} samples")

    return class_datasets


def evaluate_baseline_attack(attack_models, target_classifier,
                             baseline_classifier_path='baseline_classifier.pth',
                             hidden_dim=512, device='cpu'):
    """
    Evaluate attack on baseline classifier
    """
    print("\n" + "="*70)
    print("EVALUATING ATTACK ON BASELINE CLASSIFIER")
    print("="*70)

    checkpoint = torch.load(baseline_classifier_path, map_location=device, weights_only=False)
    target_training_indices = checkpoint['training_indices']

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    cifar10_train_dataset = datasets.CIFAR10(root='../data', train=True, download=True, transform=transform)
    cifar10_test_dataset = datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)

    member_dataset = torch.utils.data.Subset(cifar10_train_dataset, target_training_indices)
    member_loader = DataLoader(member_dataset, batch_size=128, shuffle=False)
    member_features_by_class, _ = extract_baseline_features(target_classifier, member_loader, device)

    non_member_loader = DataLoader(cifar10_test_dataset, batch_size=128, shuffle=False)
    non_member_features_by_class, _ = extract_baseline_features(target_classifier, non_member_loader, device)

    all_predictions = []
    all_true_labels = []
    all_class_labels = []

    for class_id in range(10):
        attack_model = attack_models[class_id]
        attack_model.eval()

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

        all_predictions.append(predictions_class)
        all_true_labels.append(y_test_class)
        all_class_labels.extend([class_id] * len(y_test_class))

    predictions = np.concatenate(all_predictions)
    y_test = np.concatenate(all_true_labels)

    correct = (predictions == y_test).sum()
    accuracy = 100 * correct / len(y_test)

    tp = ((predictions == 1) & (y_test == 1)).sum()
    fp = ((predictions == 1) & (y_test == 0)).sum()
    tn = ((predictions == 0) & (y_test == 0)).sum()
    fn = ((predictions == 0) & (y_test == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print("\nRESULTS:")
    print(f"  Accuracy:  {accuracy:.2f}%")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1_score:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")

    print("="*70)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score
    }


if __name__ == "__main__":
    print("This script will demonstrate that attacks WORK on baseline classifiers!")
    print("Use it to train and evaluate attacks on the vulnerable baseline.")

#!/usr/bin/env python3
"""
Complete demonstration: Membership inference attack on CNN classifier

This script demonstrates that attacks WORK on vulnerable CNN models.

Steps:
1. Train CNN classifier - vulnerable
2. Train 20 CNN shadow models
3. Train attack models on shadow data
4. Evaluate attack on CNN classifier
5. Show that attack achieves high accuracy (>60-70%)
"""

import torch
from train_cnn_classifier import train_cnn_classifier
from train_cnn_shadows import train_cnn_shadow_models
from cnn_attack import prepare_cnn_attack_dataset, evaluate_cnn_attack
from cnn_classifier_model import CNNClassifier
from membership_inference_attack import train_attack_models


def main():
    print("="*70)
    print("MEMBERSHIP INFERENCE ATTACK DEMONSTRATION")
    print("Target: CNN Classifier (CIFAR10)")
    print("="*70)
    print()
    print("This demonstration shows that attacks WORK on vulnerable models!")
    print("We expect attack accuracy > 60%")
    print("="*70)
    print()

    device = 'cpu'
    num_samples = 10000
    epochs = 100
    learning_rate = 0.001
    weight_decay = 1e-7

    # Step 1: Train CNN classifier
    print("[STEP 1/5] Training CNN classifier...")
    print(f"  Architecture: 2 Conv+Pool, FC(128), Output(10)")
    print(f"  Activation: Tanh")
    print(f"  Training samples: {num_samples}")
    print(f"  Epochs: {epochs}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Weight decay: {weight_decay}")
    print()

    cnn_model, target_indices = train_cnn_classifier(
        num_samples=num_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    print()

    # Step 2: Train CNN shadow models (fewer for faster demo)
    num_shadows = 20  # Use fewer for demonstration
    print(f"[STEP 2/5] Training {num_shadows} CNN shadow models...")
    print(f"  Each shadow: same architecture as target")
    print(f"  Training samples per shadow: {num_samples}")
    print()

    train_cnn_shadow_models(
        num_shadows=num_shadows,
        num_samples=num_samples,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device
    )
    print()

    # Step 3: Prepare attack dataset
    print("[STEP 3/5] Preparing attack dataset from CNN shadows...")
    class_datasets = prepare_cnn_attack_dataset(
        device=device
    )
    print()

    # Step 4: Train attack models
    print("[STEP 4/5] Training attack models...")
    attack_models = train_attack_models(
        class_datasets=class_datasets,
        epochs=50,
        device=device,
        save_dir='cnn_attack_models'
    )
    print()

    # Step 5: Evaluate attack
    print("[STEP 5/5] Evaluating attack on CNN classifier...")

    # Load CNN classifier
    checkpoint = torch.load('cnn_classifier.pth', map_location=device, weights_only=False)
    cnn_classifier = CNNClassifier().to(device)
    cnn_classifier.load_state_dict(checkpoint['model_state_dict'])
    cnn_classifier.eval()

    results = evaluate_cnn_attack(
        attack_models=attack_models,
        target_classifier=cnn_classifier,
        device=device
    )

    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print()
    print("CNN Classifier - Attack Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print()

    if results['accuracy'] > 60:
        print("✗ VULNERABLE TO ATTACK!")
        print("  The CNN classifier leaks membership information.")
        print("  Attack accuracy > 60% indicates privacy risk.")
    elif results['accuracy'] > 55:
        print("⚠ MODERATELY VULNERABLE")
        print("  The CNN classifier shows some privacy leakage.")
    else:
        print("✓ ATTACK FAILED")
        print("  Even the CNN shows strong privacy.")

    print()
    print("="*70)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Complete demonstration: Membership inference attack on baseline classifier

This script demonstrates that attacks WORK on vulnerable models.

Steps:
1. Train baseline classifier - vulnerable
2. Train 20 baseline shadow models
3. Train attack models on shadow data
4. Evaluate attack on baseline classifier
5. Show that attack achieves high accuracy (>60-70%)
"""

import torch
from train_baseline_classifier import train_baseline_classifier
from train_baseline_shadows import train_baseline_shadow_models
from baseline_attack import (prepare_baseline_attack_dataset,
                             evaluate_baseline_attack)
from baseline_classifier_model import BaselineClassifier
from membership_inference_attack import train_attack_models


def main():
    print("="*70)
    print("MEMBERSHIP INFERENCE ATTACK DEMONSTRATION")
    print("Target: Baseline Classifier")
    print("="*70)
    print()
    print("This demonstration shows that attacks WORK on vulnerable models!")
    print("We expect attack accuracy > 60%")
    print("="*70)
    print()

    device = 'cpu'
    hidden_dim = 512
    num_samples = 10000
    epochs = 100

    # Step 1: Train baseline classifier
    print("[STEP 1/5] Training baseline classifier...")
    print(f"  Architecture: 3072 -> {hidden_dim} -> 256 -> 10")
    print(f"  Training samples: {num_samples}")
    print(f"  Epochs: {epochs}")
    print()

    baseline_model, target_indices = train_baseline_classifier(
        num_samples=num_samples,
        hidden_dim=hidden_dim,
        epochs=epochs,
        device=device
    )
    print()

    # Step 2: Train baseline shadow models (fewer for faster demo)
    num_shadows = 20  # Use fewer for demonstration
    print(f"[STEP 2/5] Training {num_shadows} BASELINE shadow models...")
    print(f"  Each shadow: same architecture as target")
    print(f"  Training samples per shadow: {num_samples}")
    print()

    train_baseline_shadow_models(
        num_shadows=num_shadows,
        hidden_dim=hidden_dim,
        num_samples=num_samples,
        epochs=epochs,
        device=device
    )
    print()

    # Step 3: Prepare attack dataset
    print("[STEP 3/5] Preparing attack dataset from baseline shadows...")
    class_datasets = prepare_baseline_attack_dataset(
        hidden_dim=hidden_dim,
        device=device
    )
    print()

    # Step 4: Train attack models
    print("[STEP 4/5] Training attack models...")
    attack_models = train_attack_models(
        class_datasets=class_datasets,
        epochs=50,
        device=device,
        save_dir='baseline_attack_models'
    )
    print()

    # Step 5: Evaluate attack
    print("[STEP 5/5] Evaluating attack on BASELINE classifier...")

    # Load baseline classifier
    checkpoint = torch.load('baseline_classifier.pth', map_location=device, weights_only=False)
    baseline_classifier = BaselineClassifier(hidden_dim=hidden_dim).to(device)
    baseline_classifier.load_state_dict(checkpoint['model_state_dict'])
    baseline_classifier.eval()

    results = evaluate_baseline_attack(
        attack_models=attack_models,
        target_classifier=baseline_classifier,
        hidden_dim=hidden_dim,
        device=device
    )

    # Summary
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE!")
    print("="*70)
    print()
    print("Baseline Classifier - Attack Results:")
    print(f"  Accuracy:  {results['accuracy']:.2f}%")
    print(f"  Precision: {results['precision']:.4f}")
    print(f"  Recall:    {results['recall']:.4f}")
    print(f"  F1 Score:  {results['f1_score']:.4f}")
    print()

    if results['accuracy'] > 60:
        print("✗ VULNERABLE TO ATTACK!")
        print("  The baseline classifier leaks membership information.")
        print("  Attack accuracy > 60% indicates privacy risk.")
    elif results['accuracy'] > 55:
        print("⚠ MODERATELY VULNERABLE")
        print("  The baseline classifier shows some privacy leakage.")
    else:
        print("✓ ATTACK FAILED")
        print("  Even the baseline shows strong privacy.")

    print()
    print("="*70)


if __name__ == "__main__":
    main()

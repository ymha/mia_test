# Baseline MLP - Membership Inference Attack

This directory contains code for membership inference attacks on a baseline MLP (Multi-Layer Perceptron) classifier trained on CIFAR10.

## Architecture

**Target Model: Baseline MLP Classifier**
- Input: 3072 dimensions (32×32×3 CIFAR10 images flattened)
- Hidden layer 1: 512 neurons
- Hidden layer 2: 256 neurons
- Output: 10 classes
- Activation: ReLU
- **Intentionally vulnerable** to demonstrate membership inference attacks

## Files

- **baseline_classifier_model.py**: MLP model architecture
- **train_baseline_classifier.py**: Train the target MLP classifier
- **train_baseline_shadows.py**: Train shadow models for attack
- **baseline_attack.py**: Membership inference attack implementation
- **run_baseline_attack_demo.py**: Complete end-to-end demo
- **membership_inference_attack.py**: Shared attack model architecture

## Usage

### Option 1: Full Demo (Recommended)
Run the complete pipeline automatically:
```bash
cd baseline_mlp
python run_baseline_attack_demo.py
```

This will:
1. Train the target MLP classifier (10K samples, 100 epochs)
2. Train 20 shadow models
3. Prepare attack dataset from shadow models
4. Train attack models
5. Evaluate attack effectiveness

### Option 2: Step-by-Step

**Step 1: Train target classifier**
```bash
python train_baseline_classifier.py --num_samples 10000 --epochs 100 --device cpu
```

**Step 2: Train shadow models**
```bash
python train_baseline_shadows.py --num_shadows 20 --epochs 100 --device cpu
```

**Step 3-5: Run attack evaluation**
Use the functions in `baseline_attack.py` to prepare attack data and evaluate.

## Key Parameters

- **num_samples**: 10,000 (default training samples)
- **hidden_dim**: 512 (default hidden layer size)
- **epochs**: 100 (allows memorization)
- **learning_rate**: 0.001
- **device**: 'cpu' or 'cuda'

## Output Files

- `baseline_classifier.pth`: Trained target model
- `baseline_shadow_models/`: Directory with shadow models
- `baseline_attack_models/`: Directory with attack models

## Expected Results

The MLP architecture is vulnerable to membership inference attacks. Expected attack accuracy: **>60%**

This demonstrates privacy risks in machine learning models without proper protections.

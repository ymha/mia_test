# CNN Attack - Membership Inference Attack

This directory contains code for membership inference attacks on a CNN (Convolutional Neural Network) classifier trained on CIFAR10.

## Architecture

**Target Model: CNN Classifier**
- **Input**: 32×32×3 CIFAR10 RGB images
- **Conv1**: 3 → 32 channels, 3×3 kernel, Tanh activation
- **MaxPool1**: 2×2 (32×32 → 16×16)
- **Conv2**: 32 → 64 channels, 3×3 kernel, Tanh activation
- **MaxPool2**: 2×2 (16×16 → 8×8)
- **Flatten**: 64 × 8 × 8 = 4096 dimensions
- **FC1**: 4096 → 128, Tanh activation
- **FC2**: 128 → 10 (output)
- **Activation**: Tanh throughout
- **Intentionally vulnerable** to demonstrate membership inference attacks

## Training Parameters

- **Learning rate**: 0.001
- **Weight decay (learning rate decay)**: 1e-7
- **Maximum epochs**: 100
- **Batch size**: 128

## Files

- **cnn_classifier_model.py**: CNN model architecture
- **train_cnn_classifier.py**: Train the target CNN classifier
- **train_cnn_shadows.py**: Train shadow models for attack
- **cnn_attack.py**: Membership inference attack implementation
- **run_cnn_attack_demo.py**: Complete end-to-end demo
- **membership_inference_attack.py**: Shared attack model architecture

## Usage

### Option 1: Full Demo (Recommended)
Run the complete pipeline automatically:
```bash
cd cnn_attack
python run_cnn_attack_demo.py
```

This will:
1. Train the target CNN classifier (10K samples, 100 epochs)
2. Train 20 CNN shadow models
3. Prepare attack dataset from shadow models
4. Train attack models
5. Evaluate attack effectiveness

### Option 2: Step-by-Step

**Step 1: Train target CNN classifier**
```bash
python train_cnn_classifier.py --num_samples 10000 --epochs 100 --learning_rate 0.001 --weight_decay 1e-7 --device cpu
```

**Step 2: Train shadow models**
```bash
python train_cnn_shadows.py --num_shadows 20 --epochs 100 --learning_rate 0.001 --weight_decay 1e-7 --device cpu
```

**Step 3-5: Run attack evaluation**
Use the functions in `cnn_attack.py` to prepare attack data and evaluate.

## Key Parameters

- **num_samples**: 10,000 (default training samples)
- **epochs**: 100 (allows memorization)
- **learning_rate**: 0.001
- **weight_decay**: 1e-7
- **device**: 'cpu' or 'cuda'

## Output Files

- `cnn_classifier.pth`: Trained target CNN model
- `cnn_shadow_models/`: Directory with shadow models
- `cnn_attack_models/`: Directory with attack models

## Expected Results

The CNN architecture with the specified hyperparameters is vulnerable to membership inference attacks. Expected attack accuracy: **>60%**

This demonstrates that even convolutional architectures can leak membership information when trained without privacy protections.

## Comparison with Baseline MLP

This CNN implementation:
- Uses spatial structure (convolutions) instead of flattening
- Has Tanh activation instead of ReLU
- Includes specific learning rate decay parameter
- Represents a more realistic image classification architecture

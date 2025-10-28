# Membership Inference Attack - CIFAR10

This repository contains implementations of membership inference attacks on different model architectures trained on CIFAR10.

## Author

Youngmok Ha @ Imperial College London

## License

Educational and research use only.

## Directory Structure

```
mia_test/
├── mlp_attack/            # MLP-based membership inference attack
│   ├── mlp_classifier_model.py
│   ├── train_mlp_classifier.py
│   ├── train_mlp_shadows.py
│   ├── mlp_attack.py
│   ├── run_mlp_attack_demo.py
│   ├── membership_inference_attack.py
│   └── README.md
│
├── cnn_attack/            # CNN-based membership inference attack
│   ├── cnn_classifier_model.py
│   ├── train_cnn_classifier.py
│   ├── train_cnn_shadows.py
│   ├── cnn_attack.py
│   ├── run_cnn_attack_demo.py
│   ├── membership_inference_attack.py
│   └── README.md
│
├── data/                  # Shared CIFAR10 dataset (auto-downloaded)
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Two Attack Implementations

### 1. MLP Attack (`mlp_attack/`)
- **Architecture**: Multi-Layer Perceptron (3072 → 512 → 256 → 10)
- **Input**: Flattened CIFAR10 images
- **Activation**: ReLU
- **Purpose**: Baseline attack on simple fully-connected network

### 2. CNN Attack (`cnn_attack/`)
- **Architecture**: CNN with 2 Conv+Pool layers, FC(128), Output(10)
- **Input**: CIFAR10 RGB images (32×32×3)
- **Activation**: Tanh
- **Learning rate**: 0.001, Weight decay: 1e-7
- **Purpose**: Attack on convolutional neural network

## Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Run Attacks

**MLP Attack:**
```bash
cd mlp_attack
python run_mlp_attack_demo.py
```

**CNN Attack:**
```bash
cd cnn_attack
python run_cnn_attack_demo.py
```

## How Membership Inference Works

1. **Train Target Model**: Train a classifier on a subset of training data
2. **Train Shadow Models**: Train multiple models mimicking the target architecture
3. **Generate Attack Data**: Use shadow models to create (features, membership) pairs
4. **Train Attack Model**: Train a binary classifier to predict membership
5. **Evaluate Attack**: Test attack on target model to measure privacy leakage

## Attack Success Metrics

- **Accuracy >50%**: Better than random guessing
- **Accuracy >60%**: Significant privacy leakage
- **Accuracy >70%**: High vulnerability

## Key Differences Between Implementations

| Feature | MLP Attack | CNN Attack |
|---------|--------------|------------|
| Architecture | Fully Connected | Convolutional |
| Input | Flattened (3072D) | Spatial (32×32×3) |
| Activation | ReLU | Tanh |
| Parameters | ~1.6M | ~260K |
| Learning Rate Decay | No | Yes (1e-7) |

## Dataset

Both implementations use **CIFAR10**:
- Training set: 50,000 images
- Test set: 10,000 images
- Classes: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- Image size: 32×32×3 (RGB)

## Research Purpose

This code demonstrates:
- Privacy risks in machine learning models
- How models can memorize training data
- Effectiveness of membership inference attacks
- Importance of privacy-preserving techniques (DP, regularization, etc.)

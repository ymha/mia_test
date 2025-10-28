# Membership Inference Attack Methodology

**Author:** Youngmok Ha @ Imperial College London

---

## Table of Contents

1. [Overview](#overview)
2. [Background](#background)
3. [System Architecture](#system-architecture)
4. [Shadow Model Training](#shadow-model-training)
5. [Attack Model Training Data Preparation](#attack-model-training-data-preparation)
6. [Attack Model Architecture and Training](#attack-model-architecture-and-training)
7. [Attack Evaluation](#attack-evaluation)
8. [Key Implementation Details](#key-implementation-details)
9. [Results Interpretation](#results-interpretation)
10. [References](#references)

---

## Overview

This document describes the implementation of a **membership inference attack** against machine learning classifiers trained on CIFAR-10. The attack aims to determine whether a specific data sample was part of a model's training set, which represents a privacy vulnerability in machine learning systems.

The implementation demonstrates attacks on two architectures:
- **MLP (Multi-Layer Perceptron)**: Fully connected network operating on flattened images
- **CNN (Convolutional Neural Network)**: Convolutional architecture preserving spatial structure

---

## Background

### What is Membership Inference?

A membership inference attack attempts to answer the question: *"Was this specific data sample used to train this model?"*

**Why does this matter?**
- Training data may contain sensitive information (medical records, personal photos, financial data)
- Successful membership inference reveals privacy leakage
- Models can memorize training data, especially when overfitting

### The Core Insight

Machine learning models exhibit different behavior on training data versus unseen data:
- **Training samples (members)**: Higher confidence predictions, sharper probability distributions
- **Unseen samples (non-members)**: Lower confidence, more uncertain predictions

The attack exploits this overfitting signature to infer membership.

---

## System Architecture

The attack consists of three main components:

```
┌───────────────────────────────────────────────────────────────┐
│                    ATTACK PIPELINE                            │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐        ┌───────────┐  │
│  │   Target     │      │   Shadow     │        │  Attack   │  │
│  │   Model      │      │   Models     │        │  Models   │  │
│  │              │      │              │        │           │  │
│  │  (Victim)    │      │  (Attacker   │        │ (Learn    │  │
│  │              │      │   trains)    │        │  patterns)│  │
│  └──────────────┘      └──────────────┘        └───────────┘  │
│         │                      │                     │        │
│         │                      ▼                     │        │
│         │              Extract features              │        │
│         │              Create labels                 │        │
│         │              (member/non-member)           │        │
│         │                      │                     │        │
│         │                      ├────────────────────►│        │
│         │                         Training data      │        │
│         │                                            │        │
│         └─────────────────────────────────────────►  │        │
│                                  Inference on        │        │
│                                  target model        │        │
│                                                      │        │
└───────────────────────────────────────────────────────────────┘
```

### Component Roles

1. **Target Model**: The victim model being attacked (trained by victim)
2. **Shadow Models**: Surrogate models trained by attacker to mimic target's behavior
3. **Attack Models**: Binary classifiers trained to predict membership from model outputs

---

## Shadow Model Training

### Purpose

Shadow models serve as proxies for the target model, allowing the attacker to generate labeled training data for the attack model without access to the target's training data.

### Data Sampling Strategy

#### Step 1: Identify Available Data

```python
# Total CIFAR-10 training set
all_indices = [0, 1, 2, ..., 49999]  # 50,000 samples

# Target model's training data (unknown to attacker in real scenario)
target_indices = [100, 255, 387, ...]  # e.g., 10,000 samples

# Available pool for shadow models (exclude target's data)
available_indices = all_indices - target_indices  # 40,000 samples
```

**Rationale:** Shadow models must NOT train on target's data to simulate realistic attack scenarios where the attacker doesn't know the target's training set.

#### Step 2: Sample Training Data for Each Shadow

For each shadow model (indexed 0 to N-1):

```python
def train_single_shadow(shadow_id, num_samples=10000):
    # Set unique random seed
    np.random.seed(shadow_id)
    torch.manual_seed(shadow_id)

    # Randomly sample from available pool
    shadow_train_indices = np.random.choice(
        available_indices,
        num_samples,
        replace=False
    )

    # Train shadow model
    shadow_model = train_model_on_data(shadow_train_indices)

    return shadow_model, shadow_train_indices
```

**Key Properties:**
- Each shadow uses a **unique seed** → different training sets
- Shadows may **overlap** with each other (realistic)
- Shadows are **disjoint** from target (necessary for valid attack)
- Same **architecture and hyperparameters** as target (mimic behavior)

### Visual Example

```
CIFAR-10 Training Set (50,000 samples)
├───────────────────────────────────────────────────────────┤

Target Model Training Data (10,000 samples)
├──────────┤
 Indices: [100, 255, 387, ..., 9876]

Available Pool for Shadows (40,000 samples)
            ├─────────────────────────────────────────────────┤

Shadow 0 (seed=0, 10,000 samples)
            ├──────────┤
             Indices: [10000, 10532, 11234, ...]

Shadow 1 (seed=1, 10,000 samples)
                 ├──────────┤
                  Indices: [10500, 11000, 15234, ...]

Shadow 2 (seed=2, 10,000 samples)
                      ├──────────┤
                       Indices: [11234, 15000, 20123, ...]

... (continues for all shadow models)

Note: Shadows may share some samples (e.g., sample 11234 in both Shadow 0 and Shadow 2)
```

### Training Process

Each shadow model is trained identically to the target:

**MLP Classifier:**
```
Input: 32×32×3 image → Flatten to 3072-D
Architecture: 3072 → 512 → 256 → 10
Activation: ReLU
Optimizer: Adam (lr=0.001)
Epochs: 100
```

**CNN Classifier:**
```
Input: 32×32×3 image
Architecture:
  Conv1 (3→32, 3×3) → Tanh → MaxPool(2×2)
  Conv2 (32→64, 3×3) → Tanh → MaxPool(2×2)
  FC1 (4096→128) → Tanh
  FC2 (128→10)
Optimizer: Adam (lr=0.001, weight_decay=1e-7)
Epochs: 100
```

### Implementation Reference

**Files:**
- `mlp_attack/train_mlp_shadows.py`: Lines 18-176
- `cnn_attack/train_cnn_shadows.py`: Lines 18-176

**Key Functions:**
- `train_single_mlp_shadow()`: Trains one MLP shadow model
- `train_mlp_shadow_models()`: Orchestrates training of all shadows
- Similar functions exist for CNN variant

---

## Attack Model Training Data Preparation

This is the **core** of the membership inference attack. The attacker uses shadow models to create labeled examples of member vs. non-member patterns.

### High-Level Process

```
For each shadow model:
  1. Load shadow model and its training indices
  2. Create MEMBER examples: Run shadow on its training data
  3. Create NON-MEMBER examples: Run shadow on unseen data
  4. Extract features (confidence scores) from predictions
  5. Label: member=1, non-member=0
  6. Aggregate across all shadow models
  7. Balance and prepare final attack training dataset
```

### Step 1: Feature Extraction

For each data sample, extract an **11-dimensional feature vector**:

```python
def extract_features(model, data_loader):
    """
    Extract features for membership inference attack

    Returns:
        Feature vector: [class_label, prob_0, prob_1, ..., prob_9]
    """
    model.eval()
    features = []

    with torch.no_grad():
        for images, labels in data_loader:
            # Forward pass
            logits = model(images)
            probs = torch.softmax(logits, dim=1)  # Convert to probabilities

            # Create feature vector for each sample
            for i in range(len(labels)):
                feature = np.concatenate([
                    [labels[i]],  # True class label
                    probs[i]      # 10 probability scores
                ])
                features.append(feature)

    return np.array(features)
```

**Feature Vector Structure:**

```
Feature = [class_label, prob_class0, prob_class1, ..., prob_class9]
           ↑            ↑─────────────────────────────────────────↑
           │                    Model's confidence distribution
           │
           True class (0-9)

Example for a cat image (class 3):
  Member sample:     [3, 0.01, 0.02, 0.03, 0.89, 0.02, 0.01, 0.01, 0.00, 0.01, 0.00]
                                           ↑
                                      High confidence on true class

  Non-member sample: [3, 0.05, 0.08, 0.12, 0.35, 0.15, 0.10, 0.05, 0.03, 0.05, 0.02]
                                           ↑
                                      Lower confidence, more spread out
```

**Why these features?**
- **Class label**: Enables class-specific attack models
- **Probability distribution**: Captures model's confidence pattern
- **Intuition**: Overfitted models are overconfident on training data

### Step 2: Creating Member and Non-Member Labels

For each shadow model:

```python
def prepare_attack_dataset_from_shadow(shadow_model, shadow_train_indices):
    """
    Create labeled training data from one shadow model
    """
    # MEMBER examples: Shadow's training data
    member_dataset = CIFAR10[shadow_train_indices]
    member_features = extract_features(shadow_model, member_dataset)
    member_labels = np.ones(len(member_features))  # Label = 1 (IN)

    # NON-MEMBER examples: Data shadow never saw
    # Exclude: shadow's training data AND target's training data
    all_indices = np.arange(50000)
    non_member_candidates = all_indices - shadow_train_indices - target_train_indices

    # Sample equal number of non-members
    non_member_indices = np.random.choice(
        non_member_candidates,
        len(shadow_train_indices),
        replace=False
    )

    non_member_dataset = CIFAR10[non_member_indices]
    non_member_features = extract_features(shadow_model, non_member_dataset)
    non_member_labels = np.zeros(len(non_member_features))  # Label = 0 (OUT)

    return member_features, member_labels, non_member_features, non_member_labels
```

**Visual Example:**

```
Shadow Model 5 Training Data: {12345, 23456, 34567, ...}
                                  ↓
              ┌──────────────────────────────────────────┐
              │   Run Shadow 5 on image 12345            │
              │   Prediction: [3, 0.01, ..., 0.89, ...]  │
              │   Label: 1 (MEMBER)                      │
              └──────────────────────────────────────────┘

Shadow Model 5 has NOT seen: {55555, 66666, 77777, ...}
                                  ↓
              ┌──────────────────────────────────────────┐
              │   Run Shadow 5 on image 55555            │
              │   Prediction: [3, 0.05, ..., 0.35, ...]  │
              │   Label: 0 (NON-MEMBER)                  │
              └──────────────────────────────────────────┘
```

### Step 3: Aggregation Across All Shadow Models

Process all shadow models (typically 20) to accumulate data:

```python
# Initialize storage for each class
class_attack_features = {i: [] for i in range(10)}
class_attack_labels = {i: [] for i in range(10)}

# Process each shadow model
for shadow_id in range(num_shadows):
    # Load shadow model
    shadow_model = load_shadow_model(shadow_id)

    # Extract member and non-member features
    member_feats, member_labs, nonmember_feats, nonmember_labs = \
        prepare_attack_dataset_from_shadow(shadow_model, shadow_indices[shadow_id])

    # Group by class
    for class_id in range(10):
        # Add member examples for this class
        class_mask = (member_feats[:, 0] == class_id)
        class_attack_features[class_id].append(member_feats[class_mask])
        class_attack_labels[class_id].append(member_labs[class_mask])

        # Add non-member examples for this class
        class_mask = (nonmember_feats[:, 0] == class_id)
        class_attack_features[class_id].append(nonmember_feats[class_mask])
        class_attack_labels[class_id].append(nonmember_labs[class_mask])
```

**Example Aggregation for Class 3 (Cats):**

```
Shadow 0: 500 cat members + 500 cat non-members
Shadow 1: 520 cat members + 480 cat non-members
Shadow 2: 490 cat members + 510 cat non-members
Shadow 3: 510 cat members + 490 cat non-members
...
Shadow 19: 505 cat members + 495 cat non-members
──────────────────────────────────────────────────
Total: ~10,000 cat members + ~10,000 cat non-members
       = 20,000 training examples for Class 3 attack model
```

### Step 4: Class-Specific Balancing

Create balanced datasets for each class to prevent learning trivial patterns:

```python
def balance_attack_dataset(class_features, class_labels):
    """
    Balance member and non-member examples for each class
    """
    class_datasets = {}

    for class_id in range(10):
        # Combine all features and labels for this class
        X_class = np.vstack(class_features[class_id])
        y_class = np.concatenate(class_labels[class_id])

        # Count members and non-members
        n_members = np.sum(y_class == 1)
        n_non_members = np.sum(y_class == 0)
        n_balanced = min(n_members, n_non_members)

        # Sample equal numbers from each
        member_idx = np.where(y_class == 1)[0]
        nonmember_idx = np.where(y_class == 0)[0]

        np.random.seed(42 + class_id)  # Reproducibility
        balanced_member_idx = np.random.choice(member_idx, n_balanced, replace=False)
        balanced_nonmember_idx = np.random.choice(nonmember_idx, n_balanced, replace=False)

        # Combine and shuffle
        balanced_idx = np.concatenate([balanced_member_idx, balanced_nonmember_idx])
        np.random.shuffle(balanced_idx)

        X_train = X_class[balanced_idx]
        y_train = y_class[balanced_idx]

        class_datasets[class_id] = (X_train, y_train)

    return class_datasets
```

**Why Balance?**
- Prevents attack model from learning "always predict member" or "always predict non-member"
- Forces learning of actual confidence patterns
- Improves generalization to target model

### Final Output Format

```python
class_datasets = {
    0: (X_train_class0, y_train_class0),  # Shape: (20000, 11), (20000,)
    1: (X_train_class1, y_train_class1),  # Shape: (18000, 11), (18000,)
    2: (X_train_class2, y_train_class2),  # Shape: (19500, 11), (19500,)
    ...
    9: (X_train_class9, y_train_class9)   # Shape: (21000, 11), (21000,)
}

# Each X_train contains feature vectors: [class_label, prob_0, ..., prob_9]
# Each y_train contains binary labels: 0 (non-member) or 1 (member)
```

### Implementation Reference

**Files:**
- `mlp_attack/mlp_attack.py`: Lines 16-166
- `cnn_attack/cnn_attack.py`: Similar implementation

**Key Functions:**
- `extract_mlp_features()`: Extract 11-D feature vectors
- `prepare_mlp_attack_dataset()`: Complete data preparation pipeline

---

## Attack Model Architecture and Training

### Architecture

The attack model is a **binary classifier** that predicts membership from the 11-D feature vector:

```python
class AttackModel(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))      # Input: 11-D → 64-D
        x = self.dropout(x)              # Regularization
        x = self.fc2(x).squeeze(-1)      # 64-D → 1-D (logit)
        return x
```

**Architecture Diagram:**

```
Input: [class_label, prob_0, prob_1, ..., prob_9]  (11-D)
   │
   ├─► Linear(11 → 64)
   │
   ├─► ReLU Activation
   │
   ├─► Dropout(p=0.3)
   │
   ├─► Linear(64 → 1)
   │
   └─► Output: Logit (real number)
        │
        └─► Sigmoid → Probability of membership [0, 1]
```

**Design Choices:**
- **Simple architecture**: Avoids overfitting on meta-patterns
- **Dropout**: Regularization for better generalization
- **Single output**: Binary classification (member vs. non-member)

### Training Process

Train **10 separate attack models** (one per class) for specialization:

```python
def train_attack_models(class_datasets, epochs=50, batch_size=256, lr=1e-3):
    attack_models = {}

    for class_id in range(10):
        X_train, y_train = class_datasets[class_id]

        # Initialize model
        attack_model = AttackModel(input_dim=11)
        optimizer = Adam(attack_model.parameters(), lr=lr)
        criterion = BCEWithLogitsLoss()  # Binary cross-entropy

        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train)
        )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        for epoch in range(epochs):
            for features, labels in dataloader:
                optimizer.zero_grad()

                # Forward pass
                logits = attack_model(features)
                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

        attack_models[class_id] = attack_model

    return attack_models
```

**Training Hyperparameters:**
- Epochs: 50 (relatively few to avoid overfitting)
- Batch size: 256
- Learning rate: 0.001
- Optimizer: Adam
- Loss: Binary Cross-Entropy with Logits

**Why Class-Specific Models?**

Different classes may exhibit different overfitting patterns:

```
Class 0 (Airplane): Easy to classify → less overfitting
  Member:     [0, 0.02, 0.01, ..., 0.85, ...]  (confident)
  Non-member: [0, 0.05, 0.03, ..., 0.65, ...]  (less confident, but still decent)

Class 8 (Ship): Harder to classify → more overfitting
  Member:     [8, 0.05, 0.10, ..., 0.70, ...]  (moderately confident)
  Non-member: [8, 0.15, 0.20, ..., 0.25, ...]  (very uncertain)
```

Class-specific models can learn these nuanced patterns better.

### Implementation Reference

**Files:**
- `mlp_attack/membership_inference_attack.py`: Lines 10-134
- `cnn_attack/membership_inference_attack.py`: Identical implementation

**Key Functions:**
- `AttackModel`: Neural network architecture
- `train_attack_models()`: Training loop for all 10 attack models

---

## Attack Evaluation

### Evaluation Process

Test the trained attack models on the **target model** (the victim):

```python
def evaluate_attack(attack_models, target_classifier, target_train_indices):
    """
    Evaluate attack on target model
    """
    # Load CIFAR-10 datasets
    train_dataset = CIFAR10(train=True)
    test_dataset = CIFAR10(train=False)

    # Create member dataset (target's training data)
    member_dataset = Subset(train_dataset, target_train_indices)
    member_loader = DataLoader(member_dataset, batch_size=128)

    # Create non-member dataset (test set - never seen by target)
    non_member_loader = DataLoader(test_dataset, batch_size=128)

    # Extract features from target model
    member_features = extract_features(target_classifier, member_loader)
    non_member_features = extract_features(target_classifier, non_member_loader)

    # Prepare ground truth labels
    member_labels = np.ones(len(member_features))      # Label = 1
    non_member_labels = np.zeros(len(non_member_features))  # Label = 0

    # Run attack models
    predictions = []
    true_labels = []

    for class_id in range(10):
        # Get samples for this class
        member_class_mask = (member_features[:, 0] == class_id)
        non_member_class_mask = (non_member_features[:, 0] == class_id)

        X_test = np.vstack([
            member_features[member_class_mask],
            non_member_features[non_member_class_mask]
        ])
        y_test = np.concatenate([
            member_labels[member_class_mask],
            non_member_labels[non_member_class_mask]
        ])

        # Predict with attack model
        attack_model = attack_models[class_id]
        attack_model.eval()

        with torch.no_grad():
            logits = attack_model(torch.FloatTensor(X_test))
            preds = (logits > 0).long().numpy()  # Threshold at 0

        predictions.append(preds)
        true_labels.append(y_test)

    # Combine predictions across all classes
    predictions = np.concatenate(predictions)
    true_labels = np.concatenate(true_labels)

    # Compute metrics
    accuracy = (predictions == true_labels).mean()

    tp = ((predictions == 1) & (true_labels == 1)).sum()
    fp = ((predictions == 1) & (true_labels == 0)).sum()
    tn = ((predictions == 0) & (true_labels == 0)).sum()
    fn = ((predictions == 0) & (true_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn}
    }
```

### Evaluation Datasets

```
Target Model's Training Data (10,000 samples)
├──────────┤
 → These are MEMBERS (ground truth = 1)
 → Attack should predict: 1

CIFAR-10 Test Set (10,000 samples)
                    ├──────────┤
 → These are NON-MEMBERS (ground truth = 0)
 → Attack should predict: 0
```

**Important:** The test set is guaranteed to be non-members because it's a separate split from training.

### Implementation Reference

**Files:**
- `mlp_attack/mlp_attack.py`: Lines 169-258
- `cnn_attack/cnn_attack.py`: Similar implementation

**Key Functions:**
- `evaluate_mlp_attack()`: Complete evaluation pipeline
- `evaluate_cnn_attack()`: CNN variant

---

## Key Implementation Details

### 1. Random Seeds and Reproducibility

```python
# Target model
torch.manual_seed(42)
np.random.seed(42)

# Shadow models (each gets unique seed)
for shadow_id in range(num_shadows):
    torch.manual_seed(shadow_id)
    np.random.seed(shadow_id)
    # Train shadow...

# Attack dataset balancing
for class_id in range(10):
    np.random.seed(42 + class_id)
    # Balance dataset...
```

**Purpose:** Ensures reproducible experiments while maintaining diversity across shadow models.

### 2. Data Normalization

All models use consistent CIFAR-10 normalization:

```python
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

**Effect:** Scales pixel values from [0, 255] to [-1, 1].

### 3. Model Checkpointing

Shadow and target models save comprehensive state:

```python
torch.save({
    'model_state_dict': model.state_dict(),
    'hidden_dim': hidden_dim,
    'training_indices': indices,
    'epochs': epochs,
    'final_train_accuracy': train_acc,
    'final_test_accuracy': test_acc,
    # Additional metadata...
}, save_path)
```

**Purpose:** Enables attack analysis and debugging.

### 4. Class-Wise Feature Grouping

Features are organized by class throughout the pipeline:

```python
features_by_class = {
    0: np.array([[0, 0.1, 0.05, ..., 0.7], ...]),  # Airplane features
    1: np.array([[1, 0.2, 0.1, ..., 0.5], ...]),   # Automobile features
    ...
    9: np.array([[9, 0.15, 0.3, ..., 0.4], ...])   # Truck features
}
```

**Purpose:** Enables class-specific attack models and analysis.

### 5. GPU/CPU Flexibility

All training functions accept device parameter:

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Model().to(device)
data = data.to(device)
```

**Purpose:** Supports both GPU acceleration and CPU-only environments.

---

## Results Interpretation

### Attack Success Metrics

**Accuracy:**
- **>70%**: High vulnerability, significant privacy leakage
- **60-70%**: Moderate vulnerability, attack is effective
- **55-60%**: Low vulnerability, some privacy leakage
- **~50%**: Attack fails, no better than random guessing

**Confusion Matrix Analysis:**

```
                    Predicted
                 Member  Non-Member
Actual   Member    TP        FN
       Non-Member  FP        TN
```

- **High TP & TN**: Attack successfully distinguishes members from non-members
- **High FP**: Attack over-predicts membership (false alarms)
- **High FN**: Attack misses actual members (fails to detect)

### Expected Results

**MLP Architecture:**
- Expected attack accuracy: **>60%**
- Reason: Simple architecture with high capacity, prone to overfitting
- Flattened inputs lose spatial structure, rely more on memorization

**CNN Architecture:**
- Expected attack accuracy: **>60%**
- Reason: Despite better architecture, 100 epochs with no privacy protection still causes overfitting
- Tanh activation and weight decay provide some regularization, but not enough

### Factors Affecting Attack Success

1. **Target Model Overfitting**: More overfitting → Higher attack success
   - Measured by train-test accuracy gap
   - Gap >10% typically indicates high vulnerability

2. **Number of Shadow Models**: More shadows → More training data → Better attack
   - 10 shadows: Minimum for basic attack
   - 20 shadows: Good balance (used in this implementation)
   - 50+ shadows: Diminishing returns

3. **Shadow Model Quality**: How well shadows mimic target
   - Same architecture: Critical
   - Similar training procedure: Important
   - Similar dataset distribution: Necessary

4. **Attack Model Complexity**: Trade-off between capacity and overfitting
   - Too simple: Can't learn patterns
   - Too complex: Overfits to shadow models, fails on target
   - Current design (11→64→1): Good balance

---

## Complete Pipeline Example

Let's trace a single image through the entire attack pipeline:

### Setup

```
Image: CIFAR-10 training sample #12345 (a cat)
True Class: 3
Target Model: Trained on indices {12345, 5678, ...} (includes 12345)
```

### Stage 1: Target Model Training

```
Target model trains on image #12345
→ Sees it 100 times during training
→ Learns to predict class 3 with high confidence
```

### Stage 2: Shadow Model Training

```
Shadow 5: Trained on indices {..., 12345, ...} (includes 12345)
Shadow 7: Trained on indices {...} (does NOT include 12345)
```

### Stage 3: Attack Dataset Preparation

**From Shadow 5 (12345 is a member):**

```
Input: Image #12345
Shadow 5 predicts: [0.01, 0.02, 0.03, 0.92, ...]
                                         ↑ Very confident on class 3
Feature vector: [3, 0.01, 0.02, 0.03, 0.92, ...]
Label: 1 (MEMBER)
```

**From Shadow 7 (12345 is a non-member):**

```
Input: Image #12345
Shadow 7 predicts: [0.08, 0.12, 0.15, 0.45, ...]
                                         ↑ Less confident on class 3
Feature vector: [3, 0.08, 0.12, 0.15, 0.45, ...]
Label: 0 (NON-MEMBER)
```

### Stage 4: Attack Model Training

```
Class 3 Attack Model learns:
  Pattern A: High confidence (0.92) → Predict MEMBER
  Pattern B: Lower confidence (0.45) → Predict NON-MEMBER
```

### Stage 5: Attack Execution on Target

```
Input: Image #12345
Target model predicts: [0.02, 0.01, 0.05, 0.88, ...]
                                         ↑ High confidence (overfitting!)

Feature vector: [3, 0.02, 0.01, 0.05, 0.88, ...]

Class 3 Attack Model:
  Input: [3, 0.02, 0.01, 0.05, 0.88, ...]
  Output: Logit = 2.5 (> 0)
  Prediction: MEMBER ✓ (Correct!)

Ground Truth: Image #12345 WAS in target's training set
Attack Result: SUCCESS
```

---

## References

### Original Papers

1. **Shokri et al.** "Membership Inference Attacks Against Machine Learning Models" (2017)
   - IEEE Symposium on Security and Privacy
   - Introduced shadow model technique
   - Demonstrated attack on various ML models

2. **Salem et al.** "ML-Leaks: Model and Data Independent Membership Inference Attacks and Defenses on Machine Learning Models" (2019)
   - NDSS Symposium
   - Relaxed assumptions about attacker knowledge
   - Improved attack efficiency

### Implementation Notes

This implementation follows the shadow model approach from Shokri et al. with the following characteristics:

- **Attack Type**: Black-box (only requires model predictions)
- **Attacker Knowledge**: Model architecture, training algorithm (realistic assumption)
- **Attack Target**: CIFAR-10 image classifiers
- **Success Criterion**: Accuracy significantly above 50% (random guessing)

### Code Organization

```
mia_test/
├── mlp_attack/                    # MLP-based attack
│   ├── mlp_classifier_model.py    # Target/shadow model architecture
│   ├── train_mlp_classifier.py    # Train target model
│   ├── train_mlp_shadows.py       # Train shadow models
│   ├── mlp_attack.py              # Attack dataset preparation & evaluation
│   ├── membership_inference_attack.py  # Attack model architecture & training
│   ├── run_mlp_attack_demo.py     # End-to-end demo
│   └── README.md                  # MLP-specific documentation
│
├── cnn_attack/                    # CNN-based attack (similar structure)
│   └── ...
│
├── data/                          # CIFAR-10 dataset (auto-downloaded)
├── requirements.txt               # Python dependencies
├── README.md                      # Main documentation
└── METHODOLOGY.md                 # This document
```

---

## Conclusion

This membership inference attack demonstrates a fundamental privacy vulnerability in machine learning models: **models can memorize training data**, and this memorization can be detected by analyzing prediction confidence patterns.

### Key Takeaways

1. **Overfitting = Privacy Risk**: The train-test accuracy gap directly correlates with attack success

2. **Shadow Models Work**: Attackers can effectively mimic target behavior without access to training data

3. **Confidence Matters**: Probability distributions reveal more than just predictions

4. **Class-Specific Patterns**: Different classes exhibit different overfitting behaviors

5. **Simple Attacks Suffice**: A shallow neural network (11→64→1) is enough to exploit these patterns

### Defense Implications

To protect against membership inference:
- **Regularization**: Dropout, weight decay, early stopping
- **Differential Privacy**: Add noise during training (DP-SGD)
- **Model Pruning**: Remove unnecessary memorization capacity
- **Ensemble Methods**: Aggregate multiple models to smooth confidence
- **Prediction Calibration**: Post-process outputs to reduce overconfidence

This implementation serves as a baseline for evaluating privacy-preserving machine learning techniques.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-28
**Author:** Youngmok Ha @ Imperial College London

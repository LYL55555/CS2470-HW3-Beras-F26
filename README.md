# CSCI 2470 HW3 — Beras

## Overview

Implemented Beras, a simple neural network framework using NumPy, including dense layers, activation functions, loss functions, optimizers, and automatic differentiation.

The framework is used to train an MLP for MNIST handwritten digit classification.

## Model Configuration

- Architecture: 784 → 256 → 128 → 10
- Hidden activation: ReLU
- Output activation: Softmax
- Initialization: Kaiming for hidden layers, Xavier for output
- Optimizer: Adam (learning rate = 0.001)
- Loss: Categorical Cross-Entropy
- Batch size: 64
- Epochs: 10
- Random seed: 42

## Running

Activate the course environment and run:

```bash
conda activate csci2470
python assignment.py
```

This trains and evaluates the model on MNIST and saves the test predictions to `predictions.npy`.

## Results

| Metric | Value |
|---|---|
| Training Accuracy | 99.49% |
| Test Accuracy | 97.90% |
| Test Loss | 0.0955 |

The model achieved the target test accuracy of 97% within 10 epochs.
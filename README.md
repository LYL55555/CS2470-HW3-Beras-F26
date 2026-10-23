# CSCI 2470 HW3

Beras implemented with Python and NumPy for MNIST classification.

## model

- layers: 784, 256, 128, 10
- hidden activation: ReLU
- output activation: Softmax
- initialization: Kaiming for hidden layers, Xavier for output
- optimizer: Adam, learning rate 0.001
- loss: categorical cross entropy
- batch size: 64
- epochs: 10

## run

```bash
conda activate csci2470
python assignment.py
```

This trains the model and saves a new predictions.npy.

## results

- local training accuracy: 99.49%
- local test accuracy: 97.90%
- local test loss: 0.0955
- predictions.npy shape: (10000,)

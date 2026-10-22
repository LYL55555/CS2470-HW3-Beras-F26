import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []


class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        error = y_pred - y_true
        squared_error = error ** 2

        loss = np.mean(np.mean(squared_error, axis=1))
        return Tensor(loss)

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs

        batch_size = y_pred.shape[0]
        num_outputs = y_pred.shape[1]

        grad_pred = 2 * (y_pred - y_true) / (batch_size * num_outputs)
        grad_true = np.zeros_like(y_true, dtype=float)

        return [Tensor(grad_pred), Tensor(grad_true)]

class CategoricalCrossEntropy(Loss):

    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        loss = -np.sum(y_true * np.log(y_pred), axis=1)
        loss = np.mean(loss)

        return Tensor(loss)

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs

        batch_size = y_pred.shape[0]
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)

        grad_pred = -y_true / y_pred / batch_size
        grad_true = np.zeros_like(y_true, dtype=float)

        return [Tensor(grad_pred), Tensor(grad_true)]

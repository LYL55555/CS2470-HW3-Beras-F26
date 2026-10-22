import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        self.classes = np.unique(data)
        self.one_hot_map = {}

        num_classes = len(self.classes)
        one_hot_vectors = np.eye(num_classes)

        for i in range(num_classes):
            label = self.classes[i]
            self.one_hot_map[label] = one_hot_vectors[i]

    def forward(self, data):
        if not hasattr(self, "one_hot_map"):
            self.fit(data)

        num_samples = len(data)
        num_classes = len(self.classes)

        encoded = np.zeros((num_samples, num_classes))

        for i in range(num_samples):
            label = data[i]
            encoded[i] = self.one_hot_map[label]

        return encoded

    def inverse(self, data):
        decoded = []

        for row in data:
            index = np.argmax(row)
            label = self.classes[index]
            decoded.append(label)

        return np.array(decoded, dtype=self.classes.dtype)

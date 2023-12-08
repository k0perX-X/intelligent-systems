import numpy as np


class Expert:
    def __init__(self, matrix: np.matrix, sign_func_on=False):
        self.number_property, self.number_classes = matrix.shape
        self.matrix = matrix.copy()
        self.prediction_matrix = np.matrix(np.zeros(matrix.shape))
        self.sign_func_on = sign_func_on

    def train(self, func=None):
        for class_number, vector in [(i, self.matrix[:, i]) for i in range(self.number_classes)]:
            self._train_on_class(vector, class_number, func)
            # print(self.prediction_matrix)

    def _train_on_class(self, vector: np.matrix, number_of_class=-1, func=None):
        for class_number in range(self.number_classes):
            if class_number != number_of_class:
                self.prediction_matrix[:, class_number] -= vector
            else:
                self.prediction_matrix[:, class_number] += vector
            if self.sign_func_on:
                self.prediction_matrix = np.sign(self.prediction_matrix)
        if func:
            func(self)

    def predict(self, vector: np.matrix) -> np.ndarray:
        prediction = np.multiply(self.prediction_matrix, vector)
        prediction = prediction.sum(axis=0)
        return np.array(prediction).reshape(-1)

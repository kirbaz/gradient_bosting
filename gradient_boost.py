import numpy as np
import pandas as pd
from typing import Tuple
from dataclasses import dataclass
from sklearn.tree import DecisionTreeRegressor


@dataclass
class GradientBoostingRegressor:
    """Gradient boosting regressor."""
    n_estimators: int = 100
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_split: int = 2
    loss: str = 'mse'
    base_pred_: float = 0.0
    trees_ = []
    verbose: bool = False
    subsample_size: float = 0.5
    replace: bool = False

    def _mse(self, y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, np.ndarray]:
        """Mean squared error loss function and gradient."""
        loss = np.mean(np.square(y_pred - y_true))
        grad = (y_pred - y_true)
        return loss, grad

    def _subsample(self, X, y):
        y = np.array(y)
        size_ = int(self.subsample_size * X.shape[0])
        if self.replace == False:
            indices = np.random.choice(X.shape[0],
                                       size_,
                                       replace=False)
        else:
            indices = np.random.choice(X.shape[0],
                                       size_,
                                       replace=True)
        sub_X = X[indices, :]
        sub_y = y[indices]
        return sub_X, sub_y

    def fit(self, X, y):
        """Fit the model to the data."""

        self.base_pred_ = np.mean(y)
        self.trees_ = []

        # size_ = int(self.subsample_size * X.shape[0])
        # predictions = np.full(size_, self.base_pred_)
        predictions = np.full(len(y), self.base_pred_)

        if isinstance(self.loss, str):
            loss_func = getattr(self, "_" + self.loss)
        else:
            loss_func = self.loss

        for _ in range(self.n_estimators):
            tree_ = DecisionTreeRegressor(max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split)

            # sub_X, sub_y = self._subsample(X, y)
            loss, grad = loss_func(y, predictions)
            sub_X, sub_grad = self._subsample(X, grad)

            tree_.fit(sub_X, -sub_grad.reshape(-1, 1))  # передаем отрицательный градиент с измененной формой

            self.trees_.append(tree_)

            sub_predictions = self.learning_rate * tree_.predict(X)  # используем sub_X
            predictions += sub_predictions

        return self

    def predict(self, X):
        """Predict the target of new data."""
        length_ = X.shape[0]
        predictions = np.full(length_, self.base_pred_)

        for tree in self.trees_:
            predictions += self.learning_rate * tree.predict(X)

        return predictions

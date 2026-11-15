import numpy as np


class GD:
    """
    Optimizer that implements the Gradient descent algorithm.
    :ivar lr: Learning rate.
    :ivar loss: loss history.
    """

    def __init__(self, lr=0.01):
        """
        Create a GD object with learning rate and loss history.
        :param lr: Learning rate.
        """
        self.lr = lr
        self.loss = []

    def minimize(self, loss, var_list=None):
        """
        Minimize loss by updating var_list.
        :param loss: Loss function.
        :param var_list: Trainable variables.
        """
        grads_and_vars = self.compute_gradients(loss, var_list)

        var_list.clear()
        var_list.extend(self.apply_gradients(grads_and_vars))

    def compute_gradients(self, loss, var_list=None):
        """
        Compute gradients of loss on trainable variables.
        :param loss: Loss function.
        :param var_list: Trainable variables.
        :return: A list of (gradient, variable) pairs.
        """
        self.loss.append(loss(var_list))

        return list(zip(gradients(loss, var_list), var_list, strict=False))

    def apply_gradients(self, grads_and_vars):
        """
        Apply gradients to variables.
        :param grads_and_vars: List of (gradient, variable) pairs.
        :return: Representing the current iteration.
        """
        return [var - self.lr * grad for grad, var in grads_and_vars]


def gradients(ys, xs):
    """
    Calculate ∂y/∂x for each x and y in xs and ys.
    :param ys: Dependent variables.
    :param xs: Independent variables.
    :return: Gradients.
    """
    features, targets = ys.args
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)

    # Vectorized MSE gradient: -2/n * X^T (y - Xw). The previous triple Python
    # loop made every epoch O(n * d) interpreted operations.
    residuals = targets - features @ np.asarray(xs, dtype=float)

    return list(-2 * features.T @ residuals / len(targets))

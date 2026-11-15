"""Tests for the linear regression model, optimizer and metrics."""

import random

import numpy as np
import pytest
from sklearn.metrics import r2_score

from main import mean_squared_error, r2_scores
from model.linear_regression import LinearRegression
from optimizer.gradient_descent import GD, gradients


def make_dataset(n=200, seed=0):
    """Build a noiseless linear dataset with a leading-one bias column.

    Args:
        n (int): Number of samples.
        seed (int): Random seed.

    Returns:
        tuple: (features, targets, true_weights).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    x = np.hstack([np.ones((n, 1)), x])
    w = np.array([1.0, 2.0, -3.0])
    return x, x @ w, w


def test_mean_squared_error_matches_numpy():
    x, y, _ = make_dataset()
    w = [0.5, -1.0, 4.0]
    expected = np.mean((y - x @ np.array(w)) ** 2)
    assert mean_squared_error(x, y, w) == pytest.approx(expected)


def test_mean_squared_error_does_not_cancel_residuals():
    # Residuals +1 and -1 must not cancel out.
    x = [[1.0], [1.0]]
    y = [1.0, -1.0]
    assert mean_squared_error(x, y, [0.0]) == pytest.approx(1.0)


def test_r2_scores_match_sklearn_and_adjusted_formula():
    rng = np.random.default_rng(1)
    y_true = rng.normal(size=100)
    y_pred = y_true + rng.normal(scale=0.3, size=100)
    p = 3
    adjusted, r2 = r2_scores(list(y_pred), list(y_true), p)
    expected_r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    assert r2 == pytest.approx(expected_r2)
    assert adjusted == pytest.approx(1 - (1 - expected_r2) * (n - 1) / (n - p - 1))


def test_gradients_match_finite_difference():
    x, y, _ = make_dataset(n=50)
    w = [0.1, 0.2, 0.3]

    def loss(weights):
        return mean_squared_error(x, y, weights)

    from functools import partial

    analytic = gradients(partial(mean_squared_error, x, y), w)
    eps = 1e-6
    for i in range(len(w)):
        w_plus = list(w)
        w_minus = list(w)
        w_plus[i] += eps
        w_minus[i] -= eps
        numeric = (loss(w_plus) - loss(w_minus)) / (2 * eps)
        assert analytic[i] == pytest.approx(numeric, rel=1e-4)


def test_compile_default_optimizer_is_not_shared():
    a = LinearRegression(inputs=1)
    b = LinearRegression(inputs=1)
    a.compile(loss=mean_squared_error)
    b.compile(loss=mean_squared_error)
    assert a._optimizer is not b._optimizer


def test_fit_recovers_true_weights():
    random.seed(0)
    x, y, w = make_dataset()
    model = LinearRegression(inputs=3)
    model.compile(optimizer=GD(lr=0.3), loss=mean_squared_error)
    model.fit(x, y)
    assert model._weights == pytest.approx(list(w), abs=1e-4)
    assert model.evaluate(x, y) == pytest.approx(0.0, abs=1e-6)

"""Unit tests for topic soft-assignment helpers (softmax)."""

import numpy as np
from common import softmax_rows


def test_softmax_rows_sum_to_one():
    sim = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax_rows(sim, tau=0.10)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)


def test_softmax_rows_temperature_sharpens():
    sim = np.array([[1.0, 2.0, 3.0]])
    sharp = softmax_rows(sim, tau=0.05)
    soft = softmax_rows(sim, tau=1.0)
    assert sharp[0, 2] > soft[0, 2]

import numpy as np
from common import entropy_norm_from_probs, softmax_rows


def test_entropy_norm_bounds():
    rng = np.random.default_rng(42)
    sim = rng.normal(size=(5, 8))
    probs = softmax_rows(sim, tau=0.10)
    ent = entropy_norm_from_probs(probs, k=8)
    assert np.all(ent >= 0)
    assert np.all(ent <= 1 + 1e-9)


def test_softmax_rows_sum_to_one():
    sim = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    p = softmax_rows(sim, tau=0.10)
    np.testing.assert_allclose(p.sum(axis=1), 1.0)

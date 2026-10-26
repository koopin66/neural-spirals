import numpy as np
import pytest

from spiral_net import contour_segments, init_params, loss_and_grads, make_spirals, predict, train


def test_spirals_are_balanced_and_bounded():
    x, y = make_spirals(n_per_class=100)
    assert x.shape == (200, 2)
    assert (y == 0).sum() == (y == 1).sum() == 100
    assert np.abs(x).max() < 1.2


def test_backprop_matches_finite_differences():
    x, y = make_spirals(n_per_class=20)
    params = init_params(seed=1)
    _, _, grads = loss_and_grads(params, x, y)
    rng = np.random.default_rng(0)
    eps = 1e-6
    for layer, (w, b) in enumerate(params):
        for tensor, grad in ((w, grads[layer][0]), (b, grads[layer][1])):
            for _ in range(5):
                idx = tuple(rng.integers(0, n) for n in tensor.shape)
                old = tensor[idx]
                tensor[idx] = old + eps
                plus = loss_and_grads(params, x, y)[0]
                tensor[idx] = old - eps
                minus = loss_and_grads(params, x, y)[0]
                tensor[idx] = old
                assert grad[idx] == pytest.approx((plus - minus) / (2 * eps), rel=1e-4, abs=1e-8)


def test_training_separates_the_spirals():
    x, y = make_spirals()
    snaps = train(x, y)
    assert snaps[0].step == 0 and snaps[-1].step == 600
    assert snaps[-1].loss < 0.01 < snaps[0].loss
    assert snaps[-1].accuracy == 1.0
    assert np.all(np.diff([s.step for s in snaps]) > 0)
    assert np.mean((predict(snaps[-1].params, x) > 0.5) == y) == 1.0


def test_contour_of_a_circle_lies_on_the_circle():
    g = np.linspace(-1, 1, 81)
    gx, gy = np.meshgrid(g, g)
    radius = 0.6
    segs = contour_segments(np.hypot(gx, gy), g, g, level=radius)
    assert len(segs) > 50
    r = np.hypot(segs[..., 0], segs[..., 1])
    assert np.abs(r - radius).max() < 0.01


def test_contour_is_empty_without_a_crossing():
    g = np.linspace(0, 1, 10)
    assert contour_segments(np.zeros((10, 10)), g, g).shape == (0, 2, 2)

"""Two interleaved spirals and a small tanh network trained on them, in plain numpy.

Everything the animation shows is computed here, before the first frame: the data,
the network (2 -> 24 -> 24 -> 1, tanh hidden layers, a logit output), full-batch Adam
training with snapshots of the weights, and marching squares for the decision boundary.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HIDDEN = (24, 24)

Params = list[tuple[np.ndarray, np.ndarray]]  # (weights, biases) per layer


def make_spirals(n_per_class: int = 130, turns: float = 1.6, noise: float = 0.025, seed: int = 3):
    """Two spirals, one the other turned by 180 degrees; labels 0 and 1.

    Points are spread evenly along each arm (radius ~ sqrt(uniform)), with a little noise.
    """
    rng = np.random.default_rng(seed)
    t = np.sqrt(rng.uniform(0.04, 1.0, n_per_class))
    angle = t * turns * 2 * np.pi
    arm = np.column_stack([t * np.cos(angle), t * np.sin(angle)])
    x = np.vstack([arm, -arm]) + rng.normal(0, noise, (2 * n_per_class, 2))
    y = np.concatenate([np.zeros(n_per_class), np.ones(n_per_class)])
    return x, y


def init_params(seed: int = 0) -> Params:
    """LeCun-normal weights, zero biases."""
    rng = np.random.default_rng(seed)
    sizes = (2, *HIDDEN, 1)
    return [
        (rng.normal(0, 1 / np.sqrt(n_in), (n_in, n_out)), np.zeros(n_out))
        for n_in, n_out in zip(sizes[:-1], sizes[1:])
    ]


def forward(params: Params, x: np.ndarray) -> list[np.ndarray]:
    """Activations of every layer, input first; the last entry is the logit."""
    acts = [x]
    for i, (w, b) in enumerate(params):
        z = acts[-1] @ w + b
        acts.append(z if i == len(params) - 1 else np.tanh(z))
    return acts


def logit(params: Params, x: np.ndarray) -> np.ndarray:
    return forward(params, x)[-1][:, 0]


def predict(params: Params, x: np.ndarray) -> np.ndarray:
    """Probability of class 1."""
    return 1 / (1 + np.exp(-logit(params, x)))


def loss_and_grads(params: Params, x: np.ndarray, y: np.ndarray):
    """Mean binary cross-entropy, accuracy and the gradients of the loss (backprop)."""
    acts = forward(params, x)
    z = acts[-1][:, 0]
    p = 1 / (1 + np.exp(-z))
    # log(1 + e^z) - y z, written to stay finite for large |z|
    loss = np.mean(np.logaddexp(0, z) - y * z)
    acc = np.mean((p > 0.5) == y)
    delta = ((p - y) / len(y))[:, None]
    grads = []
    for i in range(len(params) - 1, -1, -1):
        w, _ = params[i]
        grads.append((acts[i].T @ delta, delta.sum(0)))
        if i > 0:
            delta = (delta @ w.T) * (1 - acts[i] ** 2)
    grads.reverse()
    return loss, acc, grads


@dataclass
class Snapshot:
    step: int
    params: Params
    loss: float
    accuracy: float


def train(
    x: np.ndarray,
    y: np.ndarray,
    steps: int = 600,
    lr: float = 0.01,
    n_snapshots: int = 180,
    seed: int = 0,
) -> list[Snapshot]:
    """Full-batch Adam on binary cross-entropy.

    Snapshots are spaced geometrically in the step number, so the fast early changes get
    as many frames of the animation as the slow late ones.
    """
    params = init_params(seed)
    m = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    v = [(np.zeros_like(w), np.zeros_like(b)) for w, b in params]
    keep = {0, *np.round(np.geomspace(1, steps, n_snapshots - 1)).astype(int).tolist()}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    snapshots = []
    for step in range(steps + 1):
        loss, acc, grads = loss_and_grads(params, x, y)
        if step in keep:
            snapshots.append(Snapshot(step, [(w.copy(), b.copy()) for w, b in params], float(loss), float(acc)))
        if step == steps:
            break
        c1, c2 = 1 - beta1 ** (step + 1), 1 - beta2 ** (step + 1)
        for i, ((w, b), (gw, gb)) in enumerate(zip(params, grads)):
            (mw, mb), (vw, vb) = m[i], v[i]
            mw, mb = beta1 * mw + (1 - beta1) * gw, beta1 * mb + (1 - beta1) * gb
            vw, vb = beta2 * vw + (1 - beta2) * gw**2, beta2 * vb + (1 - beta2) * gb**2
            m[i], v[i] = (mw, mb), (vw, vb)
            params[i] = (
                w - lr * (mw / c1) / (np.sqrt(vw / c2) + eps),
                b - lr * (mb / c1) / (np.sqrt(vb / c2) + eps),
            )
    return snapshots


def contour_segments(values: np.ndarray, xs: np.ndarray, ys: np.ndarray, level: float = 0.5) -> np.ndarray:
    """Marching squares: the segments, shape (n, 2, 2), where `values` crosses `level`.

    `values[j, i]` is the value at (xs[i], ys[j]). Crossings are placed on cell edges by
    linear interpolation; a saddle cell (all four edges crossed) gets two segments.
    """
    v = values - level
    v00, v10 = v[:-1, :-1], v[:-1, 1:]
    v01, v11 = v[1:, :-1], v[1:, 1:]
    x0, x1 = xs[:-1][None, :], xs[1:][None, :]
    y0, y1 = ys[:-1][:, None], ys[1:][:, None]
    shape = v00.shape

    def cross(va, vb, pa, pb):
        hit = (va > 0) != (vb > 0)
        t = va / np.where(hit, va - vb, 1)
        return hit, pa + t * (pb - pa)

    # the four edges of each cell: bottom, right, top, left
    edges = []
    for va, vb, (ax, ay), (bx, by) in (
        (v00, v10, (x0, y0), (x1, y0)),
        (v10, v11, (x1, y0), (x1, y1)),
        (v11, v01, (x1, y1), (x0, y1)),
        (v01, v00, (x0, y1), (x0, y0)),
    ):
        hit, px = cross(va, vb, np.broadcast_to(ax, shape), np.broadcast_to(bx, shape))
        _, py = cross(va, vb, np.broadcast_to(ay, shape), np.broadcast_to(by, shape))
        edges.append((hit, px, py))
    hits = np.stack([e[0] for e in edges], -1)
    pts = np.stack([np.stack([e[1], e[2]], -1) for e in edges], -2)
    count = hits.sum(-1)
    segs = []
    two = count == 2
    if two.any():
        idx = np.argsort(~hits[two], axis=-1, kind="stable")[:, :2]
        p = pts[two]
        rows = np.arange(len(p))
        segs.append(np.stack([p[rows, idx[:, 0]], p[rows, idx[:, 1]]], 1))
    four = count == 4
    if four.any():
        p = pts[four]
        segs.append(p[:, [0, 1]])
        segs.append(p[:, [2, 3]])
    return np.concatenate(segs) if segs else np.zeros((0, 2, 2))


if __name__ == "__main__":
    x, y = make_spirals()
    snaps = train(x, y)
    for s in snaps[:: len(snaps) // 12] + [snaps[-1]]:
        print(f"step {s.step:4d}  loss {s.loss:.4f}  accuracy {s.accuracy:.3f}")
    first = next(s.step for s in snaps if s.accuracy == 1.0)
    print(f"100% accuracy first reached at step {first}")

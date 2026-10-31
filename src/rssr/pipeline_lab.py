"""Bounded MP16 teaching pipeline; canonical Dataset/DataLoader and sample-weighted loss.

Synthetic CPU regression only; not a satellite study or checkpoint resume benchmark.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def make_loader(x, y, batch_size=2, seed=16):
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y) or not len(x):
        raise ValueError("Expected paired nonempty N-by-D sample matrices")
    if not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("Nonfinite training values")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("Positive integer batch size required")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True,
                      num_workers=0, generator=torch.Generator().manual_seed(seed))


def weighted_mean(means, sizes):
    means = np.asarray(means, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    if (means.ndim != 1 or means.shape != sizes.shape or not means.size
            or not np.isfinite(means).all() or not np.isfinite(sizes).all()
            or (sizes <= 0).any()):
        raise ValueError("Finite means and positive paired sample counts required")
    return float(np.dot(means, sizes) / sizes.sum())


def tiny_regression(seed=16, epochs=30, batch_size=2, lr=0.1):
    """Repeat one bounded CPU run twice and compare exact parameter tensors."""
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
        raise ValueError("epochs must be an integer from 1 to 50")
    if not np.isfinite(lr) or not 0 < lr <= 0.5:
        raise ValueError("Learning rate must be in (0,0.5]")
    if batch_size not in (1, 2, 3, 4, 5):
        raise ValueError("Batch size must be 1..5")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Nonnegative integer seed required")
    x = torch.linspace(-1, 1, 7, dtype=torch.float64)[:, None]
    y = 2 * x + 1
    train_x, train_y = x[:5], y[:5]
    val_x, val_y = x[5:], y[5:]

    def one_run():
        torch.manual_seed(seed)
        loader = make_loader(train_x, train_y, batch_size=batch_size, seed=seed)
        model = torch.nn.Linear(1, 1, dtype=torch.float64)
        optimizer = torch.optim.SGD(model.parameters(), lr=float(lr))
        history = []
        for epoch in range(epochs):
            model.train()
            means, sizes = [], []
            for bx, by in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = ((model(bx) - by) ** 2).mean()
                loss.backward()
                optimizer.step()
                means.append(float(loss.detach()))
                sizes.append(len(bx))
            model.eval()
            with torch.inference_mode():
                val_loss = float(((model(val_x) - val_y) ** 2).mean())
            history.append({"epoch": epoch + 1, "train_batch_weighted_MSE": weighted_mean(means, sizes),
                            "validation_MSE": val_loss})
        return {k: v.detach().clone() for k, v in model.state_dict().items()}, history, sizes

    state_a, history_a, last_batch_sizes = one_run()
    state_b, history_b, _ = one_run()
    exact = all(torch.equal(state_a[k], state_b[k]) for k in state_a)
    return {"history": history_a, "repeat_identical": exact,
            "last_batch_sizes": last_batch_sizes,
            "parameters": {k: v.numpy().tolist() for k, v in state_a.items()},
            "seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": float(lr),
            "train_count": 5, "validation_count": 2,
            "torch_version": torch.__version__, "device": "cpu",
            "task": "deterministic synthetic scalar regression; not satellite super-resolution"}

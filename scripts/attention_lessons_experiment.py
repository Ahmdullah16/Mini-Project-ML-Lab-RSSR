"""MP23–25 controlled attention lessons; train/validation only, no final-test access."""
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rssr.attention_labs import AttentionReconstruction
from rssr.experiment import load_tiles
from rssr.core import metrics

root = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)
entry = next(r for r in json.loads((root / "reports/submission-study.json").read_text()) if r["bands"] == 3)
path = root / entry["manifest_path"].replace("\\", "/")
manifest = json.loads(path.read_text())
training = load_tiles(manifest, "train", path.parent)
validation = load_tiles(manifest, "validation", path.parent)
rows, curves, outputs, gates = [], {}, {}, {}
for mode in ["off", "uniform", "channel", "spatial", "combined"]:
    torch.manual_seed(23)
    model = AttentionReconstruction(mode)
    # Every mode has identical head/tail initialization and minibatch order.
    assert model(torch.zeros(1, 3, 5, 7)).shape == (1, 3, 5, 7)
    generator = torch.Generator().manual_seed(23)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    curve = []
    start = time.perf_counter()
    for step in range(300):
        ids = torch.randint(len(training), (4,), generator=generator).tolist()
        target = torch.stack([training[i][2] for i in ids])
        optimizer.zero_grad()
        loss = F.mse_loss(model(target), target)
        loss.backward()
        optimizer.step()
        if step in (0, 49, 99, 199, 299):
            curve.append({"step": step + 1, "train_mse": loss.item()})
    train_seconds = time.perf_counter() - start
    model.eval()
    with torch.inference_mode():
        values = [{"scene_id": item["scene_id"], **metrics(model(hr[None])[0], hr, border=2)}
                  for item, _, hr in validation]
        sample = validation[0][2][None]
        outputs[mode] = model(sample)[0].permute(1, 2, 0).numpy()
        # Same learned combined features and weights are used below for all inspection modes.
        if mode == "combined":
            features = model.head(sample).relu()
            for alternative in ["off", "uniform", "channel", "spatial", "combined"]:
                gated, channel, spatial = model.attention(features, alternative)
                gates[alternative] = {"channel": channel.flatten().tolist(),
                                      "spatial": spatial[0, 0].tolist(),
                                      "feature_l2": gated.norm().item()}
            # Simple edge and texture fixtures, not learned semantic segmentation targets.
            edge = torch.zeros(1, 3, 32, 32); edge[:, :, :, 16:] = 1
            checker = ((torch.arange(32)[:, None] + torch.arange(32)[None, :]) % 2).float()
            texture = checker.expand(1, 3, -1, -1)
            for name, fixture in [("edge", edge), ("texture", texture)]:
                _, _, spatial = model.attention(model.head(fixture).relu(), "combined")
                gates[name] = {"spatial": spatial[0, 0].tolist()}
        for _ in range(5): model(sample)
        times = []
        for _ in range(20):
            t = time.perf_counter(); model(sample); times.append(time.perf_counter() - t)
    channel_n = sum(p.numel() for p in model.attention.channel.parameters())
    spatial_n = sum(p.numel() for p in model.attention.spatial.parameters())
    active_gate_n = {"off": 0, "uniform": 0, "channel": channel_n,
                     "spatial": spatial_n, "combined": channel_n + spatial_n}[mode]
    base_n = sum(p.numel() for layer in [model.head, model.tail] for p in layer.parameters())
    rows.append({"mode": mode, "active_parameters": base_n + active_gate_n,
                 "gate_parameters": active_gate_n, "allocated_parameters": sum(p.numel() for p in model.parameters()),
                 "validation_psnr": statistics.mean(v["psnr"] for v in values),
                 "per_tile": values, "training_seconds": train_seconds,
                 "median_inference_seconds": statistics.median(times)})
    curves[mode] = curve
    print(mode, rows[-1]["validation_psnr"], flush=True)

fig, axes = plt.subplots(2, 3, figsize=(10, 6), layout="constrained")
reference = validation[0][2].permute(1, 2, 0).numpy()
lo, hi = np.percentile(reference, [2, 98])
for ax, (name, array) in zip(axes.flat, [("Reference", reference), *outputs.items()]):
    ax.imshow(np.clip((array - lo) / (hi - lo), 0, 1)); ax.set_title(name); ax.axis("off")
fig.savefig(root / "reports/mp23-mp25-reconstruction.png", dpi=120); plt.close(fig)
fig, axes = plt.subplots(2, 3, figsize=(11, 6), layout="constrained")
for mode in ["off", "uniform", "channel", "combined"]:
    axes[0, 0].plot(gates[mode]["channel"], label=mode)
axes[0, 0].set(title="Same-feature channel gates", xlabel="Feature channel", ylim=(0, 1.05))
axes[0, 0].legend(fontsize=7)
for ax, name in zip([axes[0, 1], axes[0, 2], axes[1, 0], axes[1, 1]], ["spatial", "combined", "edge", "texture"]):
    panel = ax.imshow(gates[name]["spatial"], vmin=0, vmax=1, cmap="viridis")
    ax.set_title(name + " spatial gate"); fig.colorbar(panel, ax=ax)
for name, history in curves.items():
    axes[1, 2].semilogy([r["step"] for r in history], [r["train_mse"] for r in history], label=name)
axes[1, 2].set(xlabel="Step", ylabel="Training MSE"); axes[1, 2].legend(fontsize=7)
fig.savefig(root / "reports/mp23-mp25-attention.png", dpi=120); plt.close(fig)
report = {"seed": 23, "steps": 300, "batch_size": 4, "optimizer": "Adam lr=0.001", "loss": "MSE",
          "selection": "fixed final step; no validation selection", "torch": torch.__version__,
          "device": "CPU float32, two threads", "manifest_sha256": entry["manifest_sha256"],
          "data": "Four train/four validation NAIP scenes, 16 fixed 32x32 patches each; RGB stored DN/10000, not reflectance; public domain USDA/OpenSR",
          "protocol": "Same-grid reconstruction, identical initialization and batches, border=2, range=1, mean tile PSNR; equal tiles per scene",
          "results": rows, "curves": curves, "same_feature_gates": gates,
          "limitations": "One seed, small crops; gates are not explanations or semantic masks. Uniform gate is 0.5, off is 1. Allocated inactive modules preserve matched initialization but are excluded from active parameter counts. Different parameter counts despite matched steps. No held-out data, SR claim or frozen-model modification."}
(root / "reports/mp23-mp25-attention.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

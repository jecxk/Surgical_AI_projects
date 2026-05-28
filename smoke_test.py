"""Smoke test: instantiate dataset on already-processed videos, run 1 batch through model."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from src.dataset.cholec80_dataset import Cholec80Dataset, Cholec80SequenceDataset
from src.dataset.transforms import get_val_transforms

# Use a small subset that's already extracted
TEST_VIDS = [1, 2, 3]

print("=" * 60)
print("PART A — Frame-level dataset")
print("=" * 60)

frame_ds = Cholec80Dataset(
    data_root="data/cholec80",
    video_ids=TEST_VIDS,
    transform=get_val_transforms(224),
    fps=1,
)
print(f"Total frame samples: {len(frame_ds)}")
print(f"  Per video: {[len([f for f in frame_ds.frames if f['video_id'] == v]) for v in TEST_VIDS]}")

# Count actual files on disk for comparison
for v in TEST_VIDS:
    n = len(list(Path(f"data/cholec80/video{v:02d}/frames").glob("*.jpg")))
    print(f"  video{v:02d} has {n} .jpg files on disk")

if len(frame_ds) > 0:
    s = frame_ds[0]
    print(f"\nSample [0]: image={tuple(s['image'].shape)}, phase={s['phase'].item()}, "
          f"video={s['video_id']}, frame_idx={s['frame_idx']}")

print("\n" + "=" * 60)
print("PART B — Sequence dataset (the one used in train.py)")
print("=" * 60)

seq_ds = Cholec80SequenceDataset(
    data_root="data/cholec80",
    video_ids=TEST_VIDS,
    transform=get_val_transforms(224),
    sequence_length=16,
    stride=8,
    fps=1,
)
print(f"Total sequences: {len(seq_ds)}")

if len(seq_ds) > 0:
    t0 = time.time()
    s = seq_ds[0]
    print(f"Load 1 sequence took {(time.time()-t0)*1000:.0f}ms")
    print(f"Shapes: images={tuple(s['images'].shape)}, phases={tuple(s['phases'].shape)}, "
          f"tools={tuple(s['tools'].shape)}")

print("\n" + "=" * 60)
print("PART C — Quick model forward pass")
print("=" * 60)

from src.models.surgical_model import SurgicalPhaseModel
import yaml
with open("configs/default.yaml") as f:
    cfg = yaml.safe_load(f)
model_cfg = cfg["model"]
model_cfg["num_phases"] = 7
model_cfg["num_tools"] = 7

model = SurgicalPhaseModel(model_cfg)
total_p = sum(p.numel() for p in model.parameters())
print(f"Model: {model_cfg['backbone']} + {model_cfg['temporal_model']}")
print(f"Params: {total_p/1e6:.1f}M")

# Build a batch of 2 sequences (smallest possible)
if len(seq_ds) >= 2:
    batch = [seq_ds[i] for i in range(min(2, len(seq_ds)))]
    images = torch.stack([b["images"] for b in batch])  # (B, T, C, H, W)
    print(f"Batch shape: {tuple(images.shape)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    images = images.to(device)

    with torch.no_grad():
        t0 = time.time()
        out = model(images)
        dt = time.time() - t0

    if isinstance(out, dict):
        print(f"Forward pass {dt*1000:.0f}ms on {device}")
        for k, v in out.items():
            if torch.is_tensor(v):
                print(f"  {k}: {tuple(v.shape)}")
    else:
        print(f"Forward pass {dt*1000:.0f}ms on {device}")
        print(f"  output: {tuple(out.shape)}")

print("\n" + "=" * 60)
print("Smoke test DONE")
print("=" * 60)

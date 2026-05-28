"""Deep audit of extracted Cholec80 data — surface any remaining issues."""
from pathlib import Path
from collections import Counter, defaultdict
import json

ROOT = Path("data/cholec80")

# Load progress to know which videos are done
prog = json.loads(Path("data/prepare_progress.json").read_text())
done = sorted(int(v.replace("video", "")) for v in prog["completed"])
print(f"Auditing {len(done)} completed videos\n")

# Splits from config
TRAIN = list(range(1, 41))
VAL = list(range(41, 61))
TEST = list(range(61, 81))

# ---------- 1. Per-video stats ----------
print("=" * 70)
print("1. PER-VIDEO STATS (random sample)")
print("=" * 70)
print(f"{'VID':<8} {'FRAMES':>8} {'PHASE LINES':>12} {'TOOL LINES':>11} {'MIN/MAX FRAME':>16}")
print("-" * 70)

stats = {}
for v in done:
    d = ROOT / f"video{v:02d}"
    fr_files = sorted((d / "frames").glob("*.jpg"))
    fr_n = len(fr_files)
    if fr_n == 0:
        print(f"video{v:02d}    [!] NO FRAMES")
        continue
    fr_min = int(fr_files[0].stem.replace("frame_", ""))
    fr_max = int(fr_files[-1].stem.replace("frame_", ""))

    phase_lines = 0
    if (d / "phase_annotations.txt").exists():
        phase_lines = len((d / "phase_annotations.txt").read_text().splitlines()) - 1

    tool_lines = 0
    if (d / "tool_annotations.txt").exists():
        tool_lines = len((d / "tool_annotations.txt").read_text().splitlines()) - 1

    stats[v] = (fr_n, phase_lines, tool_lines, fr_min, fr_max)

# Show first 5, last 5, and any anomalies
for v in done[:5]:
    fr, p, t, mn, mx = stats[v]
    print(f"video{v:02d}    {fr:>8} {p:>12} {t:>11} {mn:>6}/{mx:<8}")
if len(done) > 10:
    print("  ...")
for v in done[-5:]:
    fr, p, t, mn, mx = stats[v]
    print(f"video{v:02d}    {fr:>8} {p:>12} {t:>11} {mn:>6}/{mx:<8}")

# ---------- 2. Frame count distribution ----------
print("\n" + "=" * 70)
print("2. VIDEO LENGTH DISTRIBUTION (frames @ 1fps = seconds)")
print("=" * 70)
fr_counts = [stats[v][0] for v in done]
print(f"Min : {min(fr_counts)} frames ({min(fr_counts)/60:.1f} min)")
print(f"Max : {max(fr_counts)} frames ({max(fr_counts)/60:.1f} min)")
print(f"Mean: {sum(fr_counts)/len(fr_counts):.0f} frames ({sum(fr_counts)/len(fr_counts)/60:.1f} min)")
print(f"Total: {sum(fr_counts)} frames = {sum(fr_counts)/3600:.1f} hours of footage")

# ---------- 3. Phase distribution across all data ----------
print("\n" + "=" * 70)
print("3. PHASE DISTRIBUTION (class imbalance check)")
print("=" * 70)

PHASE_NAMES = ["Preparation", "CalotTriDissect", "Clipping", "GallbDissect",
               "GallbPackage", "Cleaning", "GallbRetract"]

phase_counter = Counter()
phase_per_split = {"train": Counter(), "val": Counter(), "test": Counter()}

for v in done:
    d = ROOT / f"video{v:02d}"
    phase_file = d / "phase_annotations.txt"
    if not phase_file.exists():
        continue

    # Get extracted frame indices
    fr_files = sorted((d / "frames").glob("*.jpg"))
    fr_idx = set(int(f.stem.replace("frame_", "")) for f in fr_files)

    # Read phase annotations
    for line in phase_file.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2 and int(parts[0]) in fr_idx:
            ph = int(parts[1])
            phase_counter[ph] += 1
            if v in TRAIN: phase_per_split["train"][ph] += 1
            elif v in VAL: phase_per_split["val"][ph] += 1
            elif v in TEST: phase_per_split["test"][ph] += 1

total = sum(phase_counter.values())
print(f"{'PHASE':<18} {'TOTAL':>9} {'%':>6} {'TRAIN':>8} {'VAL':>8} {'TEST':>8}")
print("-" * 70)
for ph in range(7):
    c = phase_counter[ph]
    pct = c/total*100
    tr = phase_per_split["train"][ph]
    va = phase_per_split["val"][ph]
    te = phase_per_split["test"][ph]
    print(f"{PHASE_NAMES[ph]:<18} {c:>9} {pct:>5.1f}% {tr:>8} {va:>8} {te:>8}")

# Imbalance ratio
counts = [phase_counter[i] for i in range(7)]
imbalance = max(counts) / max(min(counts), 1)
print(f"\nImbalance ratio (max/min): {imbalance:.1f}x")
if imbalance > 10:
    print("[!] HIGH class imbalance - need class_weights or focal loss")

# ---------- 4. Anomaly check ----------
print("\n" + "=" * 70)
print("4. ANOMALIES")
print("=" * 70)
issues = []

# Very short videos
for v in done:
    if stats.get(v, (0,))[0] < 500:
        issues.append(f"video{v:02d}: only {stats[v][0]} frames (<500) - suspicious?")

# Phase coverage gaps in any video
for v in done:
    d = ROOT / f"video{v:02d}"
    phase_file = d / "phase_annotations.txt"
    if not phase_file.exists():
        issues.append(f"video{v:02d}: missing phase_annotations.txt")
        continue

    seen_phases = set()
    for line in phase_file.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            seen_phases.add(int(parts[1]))

    if len(seen_phases) < 5:
        issues.append(f"video{v:02d}: only {len(seen_phases)} phases ({sorted(seen_phases)})")

# Tool annotation completely missing
for v in done:
    if stats.get(v, (0,0,0))[2] == 0:
        issues.append(f"video{v:02d}: tool_annotations.txt empty or missing")

if issues:
    for i in issues[:20]:
        print(f"  [!] {i}")
    if len(issues) > 20:
        print(f"  ... and {len(issues)-20} more")
else:
    print("[OK] No anomalies detected")

# ---------- 5. Train/val/test split health ----------
print("\n" + "=" * 70)
print("5. TRAIN / VAL / TEST SPLIT")
print("=" * 70)
done_set = set(done)
tr_ok = [v for v in TRAIN if v in done_set]
va_ok = [v for v in VAL if v in done_set]
te_ok = [v for v in TEST if v in done_set]
print(f"Train videos done: {len(tr_ok)}/40")
print(f"Val videos done  : {len(va_ok)}/20")
print(f"Test videos done : {len(te_ok)}/20")

tr_frames = sum(stats.get(v, (0,))[0] for v in tr_ok)
va_frames = sum(stats.get(v, (0,))[0] for v in va_ok)
te_frames = sum(stats.get(v, (0,))[0] for v in te_ok)
print(f"\nFrames per split:")
print(f"  Train: {tr_frames}")
print(f"  Val  : {va_frames}")
print(f"  Test : {te_frames}")

"""Quick data sanity check for extracted Cholec80 videos."""
from pathlib import Path

ROOT = Path("data/cholec80")

SAMPLE_VIDS = [1, 15, 30, 45]

print(f"{'VID':<8} {'FRAMES':>8} {'FIRST':<22} {'LAST':<22} {'PHASE':<7} {'TOOL':<6}")
print("-" * 80)

for vid_id in SAMPLE_VIDS:
    vid = f"video{vid_id:02d}"
    d = ROOT / vid
    if not d.exists():
        print(f"{vid:<8} MISSING")
        continue
    fr = sorted((d / "frames").glob("*.jpg"))
    phase = (d / "phase_annotations.txt").exists()
    tool = (d / "tool_annotations.txt").exists()
    first = fr[0].name if fr else "-"
    last = fr[-1].name if fr else "-"
    p = "OK" if phase else "MISS"
    t = "OK" if tool else "MISS"
    print(f"{vid:<8} {len(fr):>8} {first:<22} {last:<22} {p:<7} {t:<6}")

# Check frame-annotation alignment on video01
print("\n--- Sample annotation content (video01) ---")
phase_file = ROOT / "video01" / "phase_annotations.txt"
if phase_file.exists():
    lines = phase_file.read_text().splitlines()
    print(f"phase_annotations.txt: {len(lines)} lines")
    print("  Head:", lines[:3])
    print("  Tail:", lines[-3:])

tool_file = ROOT / "video01" / "tool_annotations.txt"
if tool_file.exists():
    lines = tool_file.read_text().splitlines()
    print(f"tool_annotations.txt: {len(lines)} lines")
    print("  Head:", lines[:2])
    print("  Tail:", lines[-2:])

# Cross-check: every extracted frame must have a label
print("\n--- Frame vs annotation coverage (video01) ---")
fr_files = sorted((ROOT / "video01" / "frames").glob("*.jpg"))
fr_indices = set(int(f.stem.replace("frame_", "")) for f in fr_files)
print(f"Extracted frames: count={len(fr_indices)} (min={min(fr_indices)}, max={max(fr_indices)})")

def label_indices(path):
    idx = set()
    for line in path.read_text().splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) >= 2:
            idx.add(int(parts[0]))
    return idx

if phase_file.exists():
    phase_idx = label_indices(phase_file)
    missing = fr_indices - phase_idx
    cov = (len(fr_indices) - len(missing)) / len(fr_indices) * 100
    print(f"Phase labels: {len(phase_idx)} entries (per-frame at 25fps)")
    print(f"  Frames with phase label: {cov:.1f}%  {'[OK]' if cov >= 99.9 else '[!]'}")

if tool_file.exists():
    tool_idx = label_indices(tool_file)
    missing = fr_indices - tool_idx
    cov = (len(fr_indices) - len(missing)) / len(fr_indices) * 100
    print(f"Tool labels: {len(tool_idx)} entries (sampled at 1fps)")
    print(f"  Frames with tool label: {cov:.1f}%  {'[OK]' if cov >= 99.9 else '[!]'}")

# Image sanity check — open one frame to verify it's not corrupted
print("\n--- Image integrity ---")
try:
    import cv2
    sample = fr_files[len(fr_files)//2]
    img = cv2.imread(str(sample))
    if img is None:
        print(f"[!] Cannot read {sample}")
    else:
        h, w, c = img.shape
        print(f"Sample frame {sample.name}: {w}x{h}x{c}  [OK]")
except Exception as e:
    print(f"[!] {e}")

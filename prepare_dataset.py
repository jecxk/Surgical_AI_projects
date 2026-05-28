"""
Cholec80 Dataset Preparation Pipeline
Extracts and processes one video at a time to stay within disk space limits.

Usage:
    python prepare_dataset.py

Resume: just run again — completed videos are skipped automatically.
"""

import os
import sys
import json
import time
import shutil
import zipfile
import logging
from pathlib import Path
from datetime import datetime

# ── Paths ──────────────────────────────────────────────────────────────────────
ZIP_PATH   = Path("D:/cholec80.zip")
TEMP_DIR   = Path("D:/cholec80_tmp")
PROJECT    = Path(__file__).parent
OUTPUT_DIR = PROJECT / "data" / "cholec80"
LOG_JSON   = PROJECT / "data" / "prepare_progress.json"
LOG_FILE   = PROJECT / "data" / "prepare_pipeline.log"
FPS        = 1          # matches default.yaml
MIN_FREE_GB = 3.0       # abort if D: drops below this

# ── Logging ────────────────────────────────────────────────────────────────────
def setup_logging():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)

# ── Progress tracking ──────────────────────────────────────────────────────────
def load_progress():
    if LOG_JSON.exists():
        return json.loads(LOG_JSON.read_text())
    return {"started": None, "completed": [], "failed": []}

def save_progress(p):
    LOG_JSON.write_text(json.dumps(p, indent=2))

# ── Frame extraction (inlined from prepare_data.py) ───────────────────────────
def extract_frames(video_path: Path, frames_dir: Path, fps: int = 1) -> int:
    try:
        import cv2
        from tqdm import tqdm
    except ImportError as e:
        log.error(f"Missing dependency: {e}. Run: pip install opencv-python tqdm")
        return 0

    frames_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"Cannot open {video_path}")
        return 0

    video_fps   = cap.get(cv2.CAP_PROP_FPS)
    total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval    = max(1, int(video_fps / fps))
    saved       = 0
    frame_idx   = 0

    with tqdm(total=total, desc="  frames", unit="fr", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % interval == 0:
                out = frames_dir / f"frame_{frame_idx:08d}.jpg"
                import cv2 as _cv2
                _cv2.imwrite(str(out), frame)
                saved += 1
            frame_idx += 1
            pbar.update(1)

    cap.release()
    return saved

# ── Annotation helpers ────────────────────────────────────────────────────────
PHASE_MAP = {
    "Preparation": 0, "CalotTriangleDissection": 1, "ClippingCutting": 2,
    "GallbladderDissection": 3, "GallbladderPackaging": 4,
    "CleaningCoagulation": 5, "GallbladderRetraction": 6,
}

def convert_phase_annotation(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = src.read_text().splitlines()
    rows = ["Frame\tPhase"]
    for line in lines[1:]:
        parts = line.strip().split("\t")
        if len(parts) >= 2:
            phase_id = PHASE_MAP.get(parts[1].strip(), -1)
            if phase_id >= 0:
                rows.append(f"{parts[0].strip()}\t{phase_id}")
    dst.write_text("\n".join(rows))

def copy_tool_annotation(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

# ── Zip helpers ───────────────────────────────────────────────────────────────
def extract_member(zf: zipfile.ZipFile, zip_member: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(zip_member) as src, open(dest, "wb") as dst:
        shutil.copyfileobj(src, dst, length=64 * 1024 * 1024)   # 64 MB buffer
    return dest

def free_gb(path="D:/") -> float:
    return shutil.disk_usage(path).free / 1024**3

# ── Per-video processing ──────────────────────────────────────────────────────
def process_video(zf: zipfile.ZipFile, vid_id: int) -> bool:
    vid  = f"video{vid_id:02d}"
    log.info(f"[{vid_id:02d}/80] {vid}")

    frames_dir   = OUTPUT_DIR / vid / "frames"
    phase_dst    = OUTPUT_DIR / vid / "phase_annotations.txt"
    tool_dst     = OUTPUT_DIR / vid / "tool_annotations.txt"

    # ── Frames ────────────────────────────────────────────────────────────────
    existing = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
    if existing >= 10:
        log.info(f"  Frames already exist ({existing}), skipping extraction")
    else:
        free = free_gb()
        log.info(f"  D: free = {free:.1f} GB")
        if free < MIN_FREE_GB:
            log.error(f"  Only {free:.1f} GB free — aborting to protect disk")
            return False

        zip_video = f"videos/{vid}.mp4"
        tmp_video = TEMP_DIR / f"{vid}.mp4"

        log.info(f"  Extracting video from zip …")
        t0 = time.time()
        extract_member(zf, zip_video, tmp_video)
        mb = tmp_video.stat().st_size / 1e6
        log.info(f"  Got {mb:.0f} MB in {time.time()-t0:.1f}s")

        log.info(f"  Extracting frames at {FPS} fps …")
        saved = extract_frames(tmp_video, frames_dir, fps=FPS)

        if saved < 10:
            log.error(f"  Only {saved} frames saved — keeping video for retry")
            return False

        log.info(f"  {saved} frames saved — deleting temp video")
        tmp_video.unlink()
        log.info(f"  Freed {mb:.0f} MB")

    # ── Annotations ───────────────────────────────────────────────────────────
    ann_tmp = TEMP_DIR / "ann"
    ann_tmp.mkdir(parents=True, exist_ok=True)

    if not phase_dst.exists():
        src = ann_tmp / f"{vid}-phase.txt"
        try:
            extract_member(zf, f"phase_annotations/{vid}-phase.txt", src)
            convert_phase_annotation(src, phase_dst)
            log.info("  Phase annotations converted")
        except KeyError:
            log.warning(f"  No phase annotation in zip for {vid}")

    if not tool_dst.exists():
        src = ann_tmp / f"{vid}-tool.txt"
        try:
            extract_member(zf, f"tool_annotations/{vid}-tool.txt", src)
            copy_tool_annotation(src, tool_dst)
            log.info("  Tool annotations copied")
        except KeyError:
            log.warning(f"  No tool annotation in zip for {vid}")

    return True

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    setup_logging()
    progress = load_progress()

    if not ZIP_PATH.exists():
        log.error(f"Zip not found: {ZIP_PATH}")
        sys.exit(1)

    if progress["started"] is None:
        progress["started"] = datetime.now().isoformat()
        save_progress(progress)

    done    = set(progress["completed"])
    failed  = set(progress["failed"])
    pending = [i for i in range(1, 81) if f"video{i:02d}" not in done]

    log.info("=" * 60)
    log.info("Cholec80 Extract + Prepare Pipeline")
    log.info(f"ZIP      : {ZIP_PATH}  ({ZIP_PATH.stat().st_size/1e9:.2f} GB)")
    log.info(f"Output   : {OUTPUT_DIR}")
    log.info(f"Progress : {len(done)}/80 done, {len(pending)} remaining")
    log.info(f"D: free  : {free_gb():.1f} GB")
    log.info("=" * 60)

    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        for vid_id in pending:
            vid = f"video{vid_id:02d}"
            ok  = process_video(zf, vid_id)
            if ok:
                done.add(vid)
                failed.discard(vid)
                progress["completed"] = sorted(done)
                progress["failed"]    = sorted(failed)
                save_progress(progress)
                log.info(f"  ✅ {vid} done  ({len(done)}/80)")
            else:
                failed.add(vid)
                progress["failed"] = sorted(failed)
                save_progress(progress)
                log.warning(f"  ❌ {vid} failed — will retry next run")

    # ── Cleanup temp dir ──────────────────────────────────────────────────────
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        log.info("Temp directory cleaned up")

    # ── Final summary ─────────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    if len(done) == 80 and not failed:
        log.info("ALL 80 VIDEOS PROCESSED SUCCESSFULLY!")
        log.info(f"Output: {OUTPUT_DIR}")
        log.info("Deleting zip …")
        ZIP_PATH.unlink()
        log.info(f"✅ Dataset ready at: {OUTPUT_DIR}")
        log.info("   Next step:  python scripts/train.py")
    else:
        log.info(f"Done: {len(done)}/80  |  Failed: {len(failed)}")
        if failed:
            log.warning(f"Failed: {sorted(failed)}")
            log.info("Run the script again to retry failed videos.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

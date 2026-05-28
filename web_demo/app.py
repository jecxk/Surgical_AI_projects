"""Surgical Phase Recognition — Web Demo

Auto-discovers trained models in `results/`, exposes a small JSON API for the
frontend to upload a video clip and receive per-frame phase + tool predictions.
"""

import sys
import base64
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, render_template, request, jsonify
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.surgical_model import SurgicalPhaseModel
from src.dataset.transforms import get_val_transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web_demo")

APP_ROOT = Path(__file__).parent.parent
RESULTS_DIR = APP_ROOT / "results"
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024  # 8 GB upload cap (full Cholec80 clips)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Compute device: {device}")

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]
TOOL_NAMES = [
    "Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag",
]

# Friendly display names for the three main models we ship.
MODEL_LABELS = {
    "resnet50_lstm": "M1 · ResNet50 + BiLSTM",
    "efficientnet_b3_tcn": "M2 · EfficientNet-B3 + TCN",
    "swin_tiny_transformer": "M3 · Swin-Tiny + Transformer",
}


def _load_one(ckpt_path: Path) -> Optional[Dict]:
    """Build a model + transform from a checkpoint dir. Returns None on failure."""
    try:
        cfg_path = ckpt_path.parent.parent / "config.yaml"
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        else:
            cfg = torch.load(ckpt_path, map_location="cpu", weights_only=False).get("config", {})

        model_cfg = cfg.get("model", {})
        if not model_cfg:
            return None
        model_cfg["num_phases"] = 7
        model_cfg["num_tools"] = 7

        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = SurgicalPhaseModel(model_cfg)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        img_size = cfg.get("data", {}).get("img_size", 224)
        return {
            "model": model,
            "transform": get_val_transforms(img_size),
            "img_size": img_size,
            "sequence_length": int(cfg.get("data", {}).get("sequence_length", 8)),
            "val_f1": float(ckpt.get("best_metric", 0.0)),
            "epoch": int(ckpt.get("epoch", 0)),
            "params_m": sum(p.numel() for p in model.parameters()) / 1e6,
        }
    except Exception as e:
        log.warning(f"Could not load {ckpt_path}: {e}")
        return None


def discover_models() -> Dict[str, Dict]:
    """Scan results/<run>/checkpoints/best_model.pth for each main model."""
    found = {}
    for run_name in MODEL_LABELS:
        ckpt = RESULTS_DIR / run_name / "checkpoints" / "best_model.pth"
        if ckpt.exists():
            info = _load_one(ckpt)
            if info is not None:
                found[run_name] = info
                log.info(
                    f"  Loaded {run_name}  val_f1={info['val_f1']:.4f}  "
                    f"params={info['params_m']:.1f}M  epoch={info['epoch']}"
                )
    return found


log.info("Discovering trained models …")
MODELS = discover_models()
log.info(f"Active models: {list(MODELS.keys()) or 'NONE (running in demo/mock mode)'}")

# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    return render_template(
        "index.html",
        phases=PHASE_NAMES,
        tools=TOOL_NAMES,
    )


@app.route("/api/models")
def api_models():
    out = []
    for run_name, info in MODELS.items():
        out.append({
            "id": run_name,
            "label": MODEL_LABELS[run_name],
            "val_f1": round(info["val_f1"], 4),
            "params_m": round(info["params_m"], 1),
            "epoch": info["epoch"],
        })
    if len(MODELS) >= 2:
        out.append({
            "id": "__ensemble__",
            "label": f"Ensemble · weighted avg ({len(MODELS)} models)",
            "val_f1": None,
            "params_m": sum(MODELS[k]["params_m"] for k in MODELS),
            "epoch": None,
        })
    return jsonify({
        "models": out,
        "device": str(device),
        "demo_mode": len(MODELS) == 0,
        "phases": PHASE_NAMES,
        "tools": TOOL_NAMES,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "Empty filename"}), 400

    model_id = request.form.get("model", next(iter(MODELS), "__demo__"))
    try:
        num_samples = max(5, min(80, int(request.form.get("samples", 24))))
    except ValueError:
        num_samples = 24

    filepath = UPLOAD_DIR / "current_upload.mp4"
    video.save(str(filepath))

    # Fast path: if the uploaded file is a Cholec80 video we already pre-extracted
    # to disk (data/cholec80/<stem>/frames/frame_XXXXXXXX.jpg at 1fps), read those
    # JPEGs instead of decoding the H.264 stream. JPEG reads are far faster than
    # seeking + decoding an mp4, cutting analysis from ~70s to a few seconds.
    stem = Path(video.filename).stem  # e.g. "video67"
    frames_dir = APP_ROOT / "data" / "cholec80" / stem / "frames"
    disk_frames = {}  # frame_index -> jpg path
    if frames_dir.is_dir():
        for fp in frames_dir.glob("frame_*.jpg"):
            try:
                disk_frames[int(fp.stem.split("_")[1])] = fp
            except (IndexError, ValueError):
                pass

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        return jsonify({"error": "Could not open uploaded video"}), 400
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total_frames / fps if fps else 0
    if total_frames < num_samples:
        num_samples = max(1, total_frames)
    interval = max(1, total_frames // max(num_samples, 1))

    use_ensemble = model_id == "__ensemble__" and len(MODELS) >= 2
    active_keys = list(MODELS.keys()) if use_ensemble else (
        [model_id] if model_id in MODELS else []
    )

    results = []
    rng = np.random.default_rng(seed=42)
    progressive_phase = 0

    # --- Plan all frame reads up front so we can decode in a single sequential pass. ---
    # Without a plan we'd seek hundreds of times in an H.264 stream (each seek decodes
    # back to a keyframe). A single forward pass with grab()/retrieve() is ~10x faster.
    def window_start_for(sample_sec: int, sequence_length: int) -> int:
        ws = (sample_sec // sequence_length) * sequence_length
        if ws + sequence_length > duration:
            ws = max(0, int(duration) - sequence_length)
        return ws

    sample_timestamps = []
    needed_seconds = set()       # all 1-fps positions any model will read
    thumb_frame_indices = set()  # frames we also need for thumbnails
    for i in range(num_samples):
        timestamp_sec = (i * interval) / fps if fps else float(i)
        sample_timestamps.append(timestamp_sec)
        thumb_frame_indices.add(min(total_frames - 1, max(0, i * interval)))
        sample_second = min(max(int(round(timestamp_sec)), 0), max(int(duration) - 1, 0))
        for info in (MODELS[k] for k in active_keys):
            sl = info["sequence_length"]
            ws = window_start_for(sample_second, sl)
            for off in range(sl):
                needed_seconds.add(ws + off)

    needed_frame_indices = {min(total_frames - 1, max(0, int(round(s * fps)))) for s in needed_seconds}
    all_targets = sorted(needed_frame_indices | thumb_frame_indices)

    decode_t0 = time.time()
    frame_bgr_cache: Dict[int, np.ndarray] = {}

    if disk_frames:
        # Fast path: read pre-extracted JPEGs. Map each target frame index to the
        # nearest available 1fps frame on disk.
        sorted_keys = np.array(sorted(disk_frames.keys()))
        for target in all_targets:
            k = int(sorted_keys[np.abs(sorted_keys - target).argmin()])
            img = cv2.imread(str(disk_frames[k]))
            if img is not None:
                frame_bgr_cache[target] = img
        cap.release()
        decode_secs = time.time() - decode_t0
        log.info(
            f"Loaded {len(frame_bgr_cache)}/{len(all_targets)} frames from disk "
            f"({stem}) in {decode_secs:.2f}s"
        )
    else:
        # Decode strategy: cluster targets that are close together, then seek to the
        # first of each cluster and grab() through it. Seeking inside an H.264 stream
        # jumps to the nearest keyframe (~10s of frames before the seek point in
        # Cholec80 mp4s), so grabbing across long sparse gaps is wasted work. A
        # cluster gap threshold of ~250 frames matches a typical GOP, so within a
        # cluster grab() stays cheap while sparse targets get a fresh seek.
        GOP_GAP = 300  # if next target is farther than this, re-seek instead of grabbing through

        clusters: list[list[int]] = []
        for t in all_targets:
            if clusters and t - clusters[-1][-1] <= GOP_GAP:
                clusters[-1].append(t)
            else:
                clusters.append([t])

        for cluster in clusters:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cluster[0])
            pos = cluster[0]
            for target in cluster:
                while pos < target:
                    if not cap.grab():
                        pos = -1
                        break
                    pos += 1
                if pos < 0:
                    break
                if cap.grab():
                    ok, frame = cap.retrieve()
                    pos += 1
                    if ok:
                        frame_bgr_cache[target] = frame
                else:
                    break
        cap.release()
        decode_secs = time.time() - decode_t0
        log.info(
            f"Decoded {len(frame_bgr_cache)}/{len(all_targets)} target frames "
            f"in {decode_secs:.2f}s (total_frames={total_frames})"
        )

    # Build per-second RGB lookup for temporal_tensor.
    second_rgb_cache: Dict[int, np.ndarray] = {}
    for sec in needed_seconds:
        fi = min(total_frames - 1, max(0, int(round(sec * fps))))
        bgr = frame_bgr_cache.get(fi)
        if bgr is not None:
            second_rgb_cache[sec] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def temporal_tensor(info: Dict, timestamp_sec: float, fallback_rgb: np.ndarray):
        """Build an evaluation-style 1-fps sequence using the pre-decoded cache."""
        sequence_length = info["sequence_length"]
        sample_second = min(max(int(round(timestamp_sec)), 0), max(int(duration) - 1, 0))
        window_start = window_start_for(sample_second, sequence_length)
        output_index = min(sequence_length - 1, max(0, sample_second - window_start))
        sequence = []
        for offset in range(sequence_length):
            rgb = second_rgb_cache.get(window_start + offset)
            sequence.append(Image.fromarray(rgb if rgb is not None else fallback_rgb))
        tensors = [info["transform"](img) for img in sequence]
        return torch.stack(tensors, dim=0).unsqueeze(0).to(device), output_index

    for i in range(num_samples):
        timestamp_sec = sample_timestamps[i]
        thumb_fi = min(total_frames - 1, max(0, i * interval))
        frame = frame_bgr_cache.get(thumb_fi)
        if frame is None:
            # cache miss — shouldn't happen, but skip rather than crash
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not active_keys:
            # Mock — model not loaded yet. Generate plausible phase progression.
            if i > 0 and rng.random() < 0.15:
                progressive_phase = min(6, progressive_phase + 1)
            phase = progressive_phase
            confidence = float(rng.uniform(0.72, 0.97))
            phase_probs = np.full(len(PHASE_NAMES), (1.0 - confidence) / (len(PHASE_NAMES) - 1))
            phase_probs[phase] = confidence
            tool_set = set()
            if phase in (1, 2, 3):
                tool_set.update(["Grasper", "Hook"])
                if rng.random() > 0.5:
                    tool_set.add("Scissors")
            elif phase == 5:
                tool_set.update(["Irrigator", "Bipolar"])
            tools = sorted(tool_set)
            per_model_probs = None
        else:
            sample_probs = []
            sample_tool_probs = []
            for key in active_keys:
                info = MODELS[key]
                tensor, output_index = temporal_tensor(info, timestamp_sec, frame_rgb)
                with torch.no_grad():
                    out = info["model"](tensor)
                p = torch.softmax(out["phase_logits"][0, output_index], dim=0).cpu().numpy()
                sample_probs.append(p)
                if "tool_logits" in out:
                    sample_tool_probs.append(torch.sigmoid(out["tool_logits"][0, output_index]).cpu().numpy())

            if use_ensemble:
                # weight by stored val_f1 (fall back to equal weights)
                weights = np.array([max(MODELS[k]["val_f1"], 0.01) for k in active_keys])
                weights = weights / weights.sum()
                phase_probs = np.average(np.stack(sample_probs), axis=0, weights=weights)
                if sample_tool_probs:
                    tool_probs = np.average(np.stack(sample_tool_probs), axis=0, weights=weights[:len(sample_tool_probs)])
                else:
                    tool_probs = None
            else:
                phase_probs = sample_probs[0]
                tool_probs = sample_tool_probs[0] if sample_tool_probs else None

            phase = int(phase_probs.argmax())
            confidence = float(phase_probs[phase])
            tools = (
                [TOOL_NAMES[j] for j, v in enumerate(tool_probs) if v > 0.5]
                if tool_probs is not None else []
            )
            per_model_probs = {k: sample_probs[idx].tolist() for idx, k in enumerate(active_keys)}

        # Thumbnail (small)
        h, w = frame.shape[:2]
        scale = 320 / max(w, 1)
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 78])
        thumb_b64 = base64.b64encode(buf).decode("ascii")

        results.append({
            "index": i,
            "timestamp": timestamp_sec,
            "phase": PHASE_NAMES[phase],
            "phase_id": phase,
            "confidence": confidence,
            "phase_probs": phase_probs.tolist() if not isinstance(phase_probs, list) else phase_probs,
            "tools": tools,
            "frame": f"data:image/jpeg;base64,{thumb_b64}",
            "per_model": per_model_probs,
        })

    # Apply optional median temporal smoothing.
    # Tail-protection: Cholec80's last two phases (GallbladderPackaging,
    # GallbladderRetraction) are ~1 min each, so a window of 5 over sparse
    # samples flattens them out. We use a smaller window (3) and skip smoothing
    # over the last 10% of samples so short closing phases survive.
    smoothing = request.form.get("smoothing", "true").lower() in ("1", "true", "yes")
    if smoothing and len(results) >= 5:
        phases = np.array([r["phase_id"] for r in results])
        from scipy.ndimage import median_filter
        smoothing_window = 3
        smoothed = median_filter(phases, size=smoothing_window).copy()
        tail_start = max(len(results) - max(1, len(results) // 10), 0)
        smoothed[tail_start:] = phases[tail_start:]
        for r, sp in zip(results, smoothed):
            r["phase_id_smoothed"] = int(sp)
            r["phase_smoothed"] = PHASE_NAMES[int(sp)]

    # Phase distribution
    counts = [0] * 7
    for r in results:
        key = r.get("phase_id_smoothed", r["phase_id"])
        counts[key] += 1
    total = sum(counts) or 1
    distribution = [
        {"phase": PHASE_NAMES[i], "phase_id": i, "fraction": counts[i] / total, "count": counts[i]}
        for i in range(7)
    ]

    return jsonify({
        "success": True,
        "model_id": model_id,
        "model_label": MODEL_LABELS.get(model_id, "Ensemble" if use_ensemble else "Demo (mock)"),
        "filename": video.filename,
        "results": results,
        "distribution": distribution,
        "smoothing_applied": smoothing,
        "temporal_context": "Each displayed prediction uses an 8-second sequence containing that timeline point.",
        "video": {
            "duration_sec": round(duration, 2),
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "sampled_frames": len(results),
        },
        "device": str(device),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "models_loaded": list(MODELS.keys()),
        "device": str(device),
    })


if __name__ == "__main__":
    # threaded=True so inference doesn't block static asset requests
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)

"""Surgical Phase Recognition — Web Demo v2 (bilingual + causal decoders).

Updates from app.py:
  - Bilingual UI labels (Vietnamese / English).
  - Exposes the new causal decoders (causal HMM with calibrated logits) alongside
    raw argmax and the offline monotonic upper bound, so a user can pick which
    decoder runs on the uploaded video.
  - Adds /api/benchmark which serves the actual test-set numbers from
    results/causal_benchmark.json, results/boundary_breakdown.json and
    results/significance.json for the new "Benchmark" tab.

Run locally:
  python web_demo/app_v2.py
  -> http://127.0.0.1:5050
"""
from __future__ import annotations

import sys
import base64
import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, List

import cv2
import numpy as np
import torch
import yaml
from flask import Flask, render_template, request, jsonify
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.surgical_model import SurgicalPhaseModel
from src.dataset.transforms import get_val_transforms
from src.evaluation.postprocess import monotonic_decode, softmax as _softmax
from src.evaluation.causal_decode import (
    estimate_transition_matrix,
    make_causal_monotonic, make_causal_hmm,
    decode_video_causal, power_transition_matrix,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("web_demo_v2")

APP_ROOT = Path(__file__).parent.parent
RESULTS_DIR = APP_ROOT / "results"
UPLOAD_DIR = APP_ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__, template_folder=str(Path(__file__).parent / "templates"))
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024 * 1024

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Compute device: {device}")

# Bilingual phase names (Vietnamese / English).
PHASE_BILINGUAL = [
    ("Chuẩn bị", "Preparation"),
    ("Bóc tách tam giác Calot", "Calot Triangle Dissection"),
    ("Kẹp và cắt", "Clipping & Cutting"),
    ("Bóc tách túi mật", "Gallbladder Dissection"),
    ("Đóng gói túi mật", "Gallbladder Packaging"),
    ("Vệ sinh và đốt cầm máu", "Cleaning & Coagulation"),
    ("Lấy túi mật ra", "Gallbladder Retraction"),
]
TOOL_BILINGUAL = [
    ("Kẹp gắp", "Grasper"),
    ("Kẹp lưỡng cực", "Bipolar"),
    ("Móc đốt", "Hook"),
    ("Kéo", "Scissors"),
    ("Kẹp clip", "Clipper"),
    ("Hút rửa", "Irrigator"),
    ("Túi đựng", "Specimen Bag"),
]
PHASE_NAMES = [en for _, en in PHASE_BILINGUAL]
TOOL_NAMES = [en for _, en in TOOL_BILINGUAL]

MODEL_LABELS = {
    "resnet50_lstm": "M1 · ResNet50 + BiLSTM",
    "efficientnet_b3_tcn": "M2 · EfficientNet-B3 + TCN",
    "swin_tiny_transformer": "M3 · Swin-Tiny + Transformer",
}

# Bilingual decoder labels surfaced to the UI.
DECODER_OPTIONS = [
    {"id": "argmax",          "vi": "Argmax thô (cơ sở)",                     "en": "Raw argmax (baseline)"},
    {"id": "median15",        "vi": "Lọc trung vị 15 (truyền thống)",         "en": "Median-15 smoothing"},
    {"id": "causal_hmm_cal",  "vi": "Causal HMM + calibration (đề xuất)",     "en": "Causal HMM + calibration (proposed)"},
    {"id": "offline_mono",    "vi": "Monotonic offline (trần lý thuyết)",     "en": "Offline monotonic (upper bound)"},
]


def _load_one(ckpt_path: Path) -> Optional[Dict]:
    try:
        cfg_path = ckpt_path.parent.parent / "config.yaml"
        if cfg_path.exists():
            cfg = yaml.safe_load(open(cfg_path, "r", encoding="utf-8"))
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
    found = {}
    for run_name in MODEL_LABELS:
        ckpt = RESULTS_DIR / run_name / "checkpoints" / "best_model.pth"
        if ckpt.exists():
            info = _load_one(ckpt)
            if info is not None:
                found[run_name] = info
                log.info(f"  Loaded {run_name}  val_f1={info['val_f1']:.4f}")
    return found


log.info("Discovering trained models …")
MODELS = discover_models()
log.info(f"Active models: {list(MODELS.keys()) or 'NONE (running in demo/mock mode)'}")


# Load benchmark artifacts (created by scripts/eval_causal.py + eval_boundary.py + eval_significance.py)
def _load_json(p: Path):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        log.warning(f"Failed to read {p}: {e}")
        return None


BENCH_DATA = _load_json(RESULTS_DIR / "causal_benchmark.json")
BOUNDARY_DATA = _load_json(RESULTS_DIR / "boundary_breakdown.json")
SIGNIFICANCE_DATA = _load_json(RESULTS_DIR / "significance.json")

# Pre-compute the learned transition matrix once if benchmark file has it
_LEARNED_A = None
if BENCH_DATA and "transition_matrix_learned" in BENCH_DATA:
    _LEARNED_A = np.array(BENCH_DATA["transition_matrix_learned"])

# Temperatures from val calibration
_TEMPS = (BENCH_DATA or {}).get("temperatures", {})


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


@app.route("/")
def index():
    return render_template(
        "index_v2.html",
        phases_vi=[vi for vi, _ in PHASE_BILINGUAL],
        phases_en=PHASE_NAMES,
        tools_vi=[vi for vi, _ in TOOL_BILINGUAL],
        tools_en=TOOL_NAMES,
        decoders=DECODER_OPTIONS,
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
            "temperature": _TEMPS.get({
                "resnet50_lstm": "M1_resnet_lstm",
                "efficientnet_b3_tcn": "M2_effnet_tcn",
                "swin_tiny_transformer": "M3_swin_transformer",
            }.get(run_name, ""), 1.0),
        })
    return jsonify({
        "models": out,
        "device": str(device),
        "demo_mode": len(MODELS) == 0,
        "decoders": DECODER_OPTIONS,
        "phases_vi": [vi for vi, _ in PHASE_BILINGUAL],
        "phases_en": PHASE_NAMES,
        "tools_vi": [vi for vi, _ in TOOL_BILINGUAL],
        "tools_en": TOOL_NAMES,
    })


@app.route("/api/benchmark")
def api_benchmark():
    """Return the actual test-set numbers for the Benchmark tab."""
    return jsonify({
        "benchmark": BENCH_DATA,
        "boundary": BOUNDARY_DATA,
        "significance": SIGNIFICANCE_DATA,
        "phases_vi": [vi for vi, _ in PHASE_BILINGUAL],
        "phases_en": PHASE_NAMES,
    })


def _apply_decoder(logits: np.ndarray, decoder_id: str, temperature: float = 1.0,
                   seconds_per_sample: float = 1.0) -> np.ndarray:
    """Apply one of the named decoders to a [T, 7] logit sequence.

    `seconds_per_sample` is the real time between consecutive rows of `logits`.
    The learned transition matrix is per-second, so when the demo sub-samples
    the video (e.g. one row every 10 s) we raise A to that power so the decoder
    is not pathologically sticky. See power_transition_matrix().
    """
    if decoder_id == "argmax":
        return logits.argmax(1)
    if decoder_id == "median15":
        from scipy.ndimage import median_filter
        # window scaled so it still covers ~15 s of real time, min 3
        win = max(3, int(round(15 / max(seconds_per_sample, 1e-6))))
        if win % 2 == 0:
            win += 1
        return median_filter(logits.argmax(1).astype(int), size=win, mode="nearest")
    if decoder_id == "offline_mono":
        return monotonic_decode(_softmax(logits))
    if decoder_id == "causal_hmm_cal":
        if _LEARNED_A is None:
            # fall back to monotonic prior if benchmark artefact missing
            dec = make_causal_monotonic(stay=0.95, temperature=temperature)
        else:
            k = max(1, int(round(seconds_per_sample)))
            A_eff = power_transition_matrix(_LEARNED_A, k) if k > 1 else _LEARNED_A
            # In the demo we additionally damp single-frame emission spikes with a
            # causal EMA (alpha=0.3) so the displayed timeline is less jittery.
            # This is a demo-display choice; the paper benchmark uses ema_alpha=1
            # (off), so reported numbers are unaffected.
            #
            # Scale EMA strength with sampling density: with dense samples (1 s
            # apart) a stronger EMA (alpha~0.3) usefully damps jitter, but with
            # sparse samples (several seconds apart) the same EMA blurs short
            # closing phases (Cleaning, Retraction) out of existence. We therefore
            # let alpha approach 1 (EMA off) as the spacing grows: alpha = min(1,
            # 0.3 * seconds_per_sample) keeps ~0.3 at 1 fps and switches EMA off
            # by ~3 s spacing, preserving short phases on sparse runs.
            ema = min(1.0, 0.3 * max(seconds_per_sample, 1.0))
            dec = make_causal_hmm(A_eff, temperature=temperature, ema_alpha=ema)
        return decode_video_causal(dec, logits)
    # default
    return logits.argmax(1)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    if "video" not in request.files:
        return jsonify({"error": "No video file in request"}), 400

    video = request.files["video"]
    if not video.filename:
        return jsonify({"error": "Empty filename"}), 400

    model_id = request.form.get("model", next(iter(MODELS), "__demo__"))
    decoder_id = request.form.get("decoder", "causal_hmm_cal")
    full_fps = request.form.get("full_fps", "false").lower() in ("1", "true", "yes")
    try:
        num_samples = max(5, min(3000, int(request.form.get("samples", 48))))
    except ValueError:
        num_samples = 48

    filepath = UPLOAD_DIR / "current_upload_v2.mp4"
    video.save(str(filepath))

    # Try the pre-extracted JPEG fast path used by app.py.
    stem = Path(video.filename).stem
    frames_dir = APP_ROOT / "data" / "cholec80" / stem / "frames"
    disk_frames = {}
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
    # "Full 1 fps" mode: one sample per second of video, matching the benchmark
    # protocol so the demo and the paper numbers use the same sampling density.
    # This is what guarantees short phases (Cleaning, Retraction) are not skipped.
    if full_fps and duration > 0:
        num_samples = max(5, min(3000, int(round(duration))))
    if total_frames < num_samples:
        num_samples = max(1, total_frames)
    interval = max(1, total_frames // max(num_samples, 1))

    active_key = model_id if model_id in MODELS else (next(iter(MODELS)) if MODELS else None)
    if active_key is None:
        cap.release()
        return jsonify({"error": "No model loaded. Train a model first."}), 503
    info = MODELS[active_key]
    seq_len = info["sequence_length"]

    # Collect all frame indices we need. Sliding causal window ending at ss.
    sample_timestamps = []
    needed_seconds = set()
    thumb_frame_indices = set()
    for i in range(num_samples):
        ts = (i * interval) / fps if fps else float(i)
        sample_timestamps.append(ts)
        thumb_frame_indices.add(min(total_frames - 1, max(0, i * interval)))
        ss = min(max(int(round(ts)), 0), max(int(duration) - 1, 0))
        ws = max(0, ss - seq_len + 1)
        for off in range(seq_len):
            needed_seconds.add(ws + off)

    needed_frame_indices = {min(total_frames - 1, max(0, int(round(s * fps)))) for s in needed_seconds}
    all_targets = sorted(needed_frame_indices | thumb_frame_indices)

    frame_bgr_cache: Dict[int, np.ndarray] = {}
    decode_t0 = time.time()
    if disk_frames:
        sorted_keys = np.array(sorted(disk_frames.keys()))
        for target in all_targets:
            k = int(sorted_keys[np.abs(sorted_keys - target).argmin()])
            img = cv2.imread(str(disk_frames[k]))
            if img is not None:
                frame_bgr_cache[target] = img
        cap.release()
    else:
        GOP_GAP = 300
        clusters: List[List[int]] = []
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
    log.info(f"Loaded {len(frame_bgr_cache)}/{len(all_targets)} frames in {decode_secs:.2f}s")

    # Build per-second RGB lookup.
    second_rgb_cache: Dict[int, np.ndarray] = {}
    for sec in needed_seconds:
        fi = min(total_frames - 1, max(0, int(round(sec * fps))))
        bgr = frame_bgr_cache.get(fi)
        if bgr is not None:
            second_rgb_cache[sec] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # Run forward pass — collect logits per sample.
    # Use a SLIDING causal window that ends at the sample second (window =
    # [ss-seq_len+1 .. ss]). Previously the window was snapped to a fixed grid
    # ((ss//seq_len)*seq_len), so two adjacent samples could fall in different
    # grid cells and receive discontinuous context, producing spurious phase
    # jumps in the timeline. A window that always ends at the current second is
    # both smoother and strictly causal (it never peeks at future frames).
    all_logits = []
    all_tool_probs = []
    inference_t0 = time.time()
    for i, ts in enumerate(sample_timestamps):
        ss = min(max(int(round(ts)), 0), max(int(duration) - 1, 0))
        ws = max(0, ss - seq_len + 1)          # window starts seq_len-1 before ss
        out_idx = min(seq_len - 1, ss - ws)    # the sample is the last valid frame
        seq_imgs = []
        fallback_rgb = None
        thumb_fi = min(total_frames - 1, max(0, i * interval))
        if thumb_fi in frame_bgr_cache:
            fallback_rgb = cv2.cvtColor(frame_bgr_cache[thumb_fi], cv2.COLOR_BGR2RGB)
        for off in range(seq_len):
            rgb = second_rgb_cache.get(ws + off, fallback_rgb)
            if rgb is None:
                rgb = np.zeros((info["img_size"], info["img_size"], 3), dtype=np.uint8)
            seq_imgs.append(Image.fromarray(rgb))
        tensors = [info["transform"](im) for im in seq_imgs]
        tensor = torch.stack(tensors, dim=0).unsqueeze(0).to(device)
        with torch.no_grad():
            o = info["model"](tensor)
        logits = o["phase_logits"][0, out_idx].cpu().numpy()
        all_logits.append(logits)
        if "tool_logits" in o:
            all_tool_probs.append(torch.sigmoid(o["tool_logits"][0, out_idx]).cpu().numpy())
    inference_secs = time.time() - inference_t0

    logits_arr = np.stack(all_logits)  # [N, 7]

    # Apply selected decoder.
    temp = _TEMPS.get({
        "resnet50_lstm": "M1_resnet_lstm",
        "efficientnet_b3_tcn": "M2_effnet_tcn",
        "swin_tiny_transformer": "M3_swin_transformer",
    }.get(active_key, ""), 1.0)
    # Real seconds between consecutive samples (interval frames / fps). Used to
    # adjust the per-second transition matrix to the sub-sampling rate.
    seconds_per_sample = max(1.0, (interval / fps) if fps else 1.0)
    decoder_t0 = time.time()
    preds = _apply_decoder(logits_arr, decoder_id, temperature=temp,
                           seconds_per_sample=seconds_per_sample)
    decoder_secs = time.time() - decoder_t0
    decoder_ms_per_frame = (decoder_secs / max(len(preds), 1)) * 1000.0

    # Also compute raw-argmax predictions for side-by-side comparison.
    preds_raw = logits_arr.argmax(1)

    # Build results.
    probs_arr = np.exp(logits_arr - logits_arr.max(axis=1, keepdims=True))
    probs_arr /= probs_arr.sum(axis=1, keepdims=True)
    results = []
    for i in range(len(preds)):
        ts = sample_timestamps[i]
        thumb_fi = min(total_frames - 1, max(0, i * interval))
        frame = frame_bgr_cache.get(thumb_fi)
        if frame is None:
            continue
        h, w = frame.shape[:2]
        scale = 320 / max(w, 1)
        thumb = cv2.resize(frame, (int(w * scale), int(h * scale)))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 78])
        thumb_b64 = base64.b64encode(buf).decode("ascii")
        phase_id = int(preds[i])
        tools = []
        if all_tool_probs:
            tp = all_tool_probs[i]
            tools = [TOOL_NAMES[j] for j, v in enumerate(tp) if v > 0.5]
        results.append({
            "index": i,
            "timestamp": ts,
            "phase_id": phase_id,
            "phase_en": PHASE_NAMES[phase_id],
            "phase_vi": PHASE_BILINGUAL[phase_id][0],
            "phase_id_raw": int(preds_raw[i]),
            "phase_en_raw": PHASE_NAMES[int(preds_raw[i])],
            "phase_vi_raw": PHASE_BILINGUAL[int(preds_raw[i])][0],
            "confidence": float(probs_arr[i, phase_id]),
            "phase_probs": probs_arr[i].tolist(),
            "tools": tools,
            "frame": f"data:image/jpeg;base64,{thumb_b64}",
        })

    # Phase distribution from decoded predictions.
    counts = [0] * 7
    for r in results:
        counts[r["phase_id"]] += 1
    total = sum(counts) or 1
    distribution = [{
        "phase_id": i,
        "phase_vi": PHASE_BILINGUAL[i][0],
        "phase_en": PHASE_NAMES[i],
        "fraction": counts[i] / total,
        "count": counts[i],
    } for i in range(7)]

    return jsonify({
        "success": True,
        "model_id": active_key,
        "model_label": MODEL_LABELS[active_key],
        "decoder_id": decoder_id,
        "decoder_label": next((d for d in DECODER_OPTIONS if d["id"] == decoder_id), DECODER_OPTIONS[0]),
        "temperature": temp,
        "filename": video.filename,
        "results": results,
        "distribution": distribution,
        "video": {
            "duration_sec": round(duration, 2),
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "sampled_frames": len(results),
        },
        "timings": {
            "decode_video_sec": round(decode_secs, 3),
            "model_inference_sec": round(inference_secs, 3),
            "decoder_total_sec": round(decoder_secs, 6),
            "decoder_ms_per_frame": round(decoder_ms_per_frame, 4),
        },
        "device": str(device),
    })


@app.route("/api/health")
def api_health():
    return jsonify({
        "ok": True,
        "models_loaded": list(MODELS.keys()),
        "device": str(device),
        "benchmark_loaded": BENCH_DATA is not None,
        "boundary_loaded": BOUNDARY_DATA is not None,
        "significance_loaded": SIGNIFICANCE_DATA is not None,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)

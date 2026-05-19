"""
Data Download & Preparation Script for Cholec80.

Note: Cholec80 requires registration at:
https://camma.unistra.fr/datasets/

This script helps prepare the data once downloaded.

Usage:
    python data/prepare_data.py --video_dir /path/to/cholec80/videos --output_dir data/cholec80
"""

import os
import sys
import argparse
import csv
from pathlib import Path
from tqdm import tqdm

try:
    import cv2
except ImportError:
    print("OpenCV not installed. Run: pip install opencv-python")
    sys.exit(1)


def extract_frames(video_path: str, output_dir: str, fps: int = 1):
    """Extract frames from a video file at the specified fps."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open {video_path}")
        return
    
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = max(1, int(video_fps / fps))
    
    print(f"  Video FPS: {video_fps}, Total frames: {total_frames}")
    print(f"  Sampling every {sample_interval} frames ({fps} fps)")
    
    frame_count = 0
    saved_count = 0
    
    pbar = tqdm(total=total_frames, desc="Extracting")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % sample_interval == 0:
            frame_path = output_dir / f"frame_{frame_count:08d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            saved_count += 1
        
        frame_count += 1
        pbar.update(1)
    
    pbar.close()
    cap.release()
    print(f"  Saved {saved_count} frames to {output_dir}")


def convert_annotations(annotation_path: str, output_path: str):
    """Convert Cholec80 phase annotations to our format."""
    phase_mapping = {
        'Preparation': 0,
        'CalotTriangleDissection': 1,
        'ClippingCutting': 2,
        'GallbladderDissection': 3,
        'GallbladderPackaging': 4,
        'CleaningCoagulation': 5,
        'GallbladderRetraction': 6,
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
    
    with open(output_path, 'w') as f:
        f.write("Frame\tPhase\n")
        for line in lines[1:]:  # Skip header
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                frame_idx = parts[0].strip()
                phase_name = parts[1].strip()
                phase_id = phase_mapping.get(phase_name, -1)
                if phase_id >= 0:
                    f.write(f"{frame_idx}\t{phase_id}\n")


def prepare_cholec80(video_dir: str, annotation_dir: str, output_dir: str, fps: int = 1):
    """Prepare Cholec80 dataset: extract frames and convert annotations."""
    video_dir = Path(video_dir)
    annotation_dir = Path(annotation_dir) if annotation_dir else video_dir
    output_dir = Path(output_dir)
    
    print("=" * 60)
    print("Cholec80 Data Preparation")
    print("=" * 60)
    
    for vid_id in range(1, 81):
        vid_name = f"video{vid_id:02d}"
        print(f"\nProcessing {vid_name}...")
        
        # Find video file
        video_path = None
        for ext in ['.mp4', '.avi', '.mkv']:
            candidate = video_dir / f"{vid_name}{ext}"
            if candidate.exists():
                video_path = candidate
                break
        
        if video_path is None:
            print(f"  ⚠ Video not found for {vid_name}, skipping")
            continue
        
        # Extract frames
        frames_dir = output_dir / vid_name / "frames"
        if not frames_dir.exists() or len(list(frames_dir.glob("*.jpg"))) == 0:
            extract_frames(str(video_path), str(frames_dir), fps=fps)
        else:
            print(f"  Frames already extracted ({len(list(frames_dir.glob('*.jpg')))} frames)")
        
        # Convert phase annotations
        phase_ann_src = annotation_dir / f"{vid_name}-phase.txt"
        phase_ann_dst = output_dir / vid_name / "phase_annotations.txt"
        if phase_ann_src.exists() and not phase_ann_dst.exists():
            convert_annotations(str(phase_ann_src), str(phase_ann_dst))
            print(f"  Phase annotations converted")
        
        # Copy tool annotations
        tool_ann_src = annotation_dir / f"{vid_name}-tool.txt"
        tool_ann_dst = output_dir / vid_name / "tool_annotations.txt"
        if tool_ann_src.exists() and not tool_ann_dst.exists():
            import shutil
            shutil.copy2(str(tool_ann_src), str(tool_ann_dst))
            print(f"  Tool annotations copied")
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print(f"   Output: {output_dir}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Prepare Cholec80 Dataset")
    parser.add_argument('--video_dir', type=str, required=True, help='Directory with Cholec80 videos')
    parser.add_argument('--annotation_dir', type=str, default=None, help='Directory with annotations')
    parser.add_argument('--output_dir', type=str, default='data/cholec80', help='Output directory')
    parser.add_argument('--fps', type=int, default=1, help='Frames per second to extract')
    args = parser.parse_args()
    
    prepare_cholec80(args.video_dir, args.annotation_dir, args.output_dir, args.fps)


if __name__ == '__main__':
    main()

"""
Cholec80 Dataset for Surgical Phase Recognition.

Handles both frame-level and sequence-level data loading for
cholecystectomy surgical videos with phase and tool annotations.
"""

import os
import json
import random
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


# Cholec80 phase definitions
PHASE_NAMES = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

# Cholec80 tool definitions
TOOL_NAMES = [
    "Grasper",
    "Bipolar",
    "Hook",
    "Scissors",
    "Clipper",
    "Irrigator",
    "SpecimenBag",
]

NUM_PHASES = 7
NUM_TOOLS = 7


class Cholec80Dataset(Dataset):
    """
    Cholec80 Frame-Level Dataset.
    
    Loads individual frames with their corresponding phase labels
    and optional tool annotations. Used for training the spatial
    feature extractor (backbone).
    
    Args:
        data_root: Root directory containing video frames and annotations
        video_ids: List of video IDs to include (1-80)
        transform: Image transforms to apply
        fps: Frames per second to sample (default: 1)
        include_tools: Whether to load tool annotations
    """
    
    def __init__(
        self,
        data_root: str,
        video_ids: List[int],
        transform=None,
        fps: int = 1,
        include_tools: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.video_ids = video_ids
        self.transform = transform
        self.fps = fps
        self.include_tools = include_tools
        
        self.frames = []  # List of (frame_path, phase_label, tool_labels)
        self._load_annotations()
    
    def _load_annotations(self):
        """Load frame paths and annotations for all specified videos."""
        for vid_id in self.video_ids:
            vid_name = f"video{vid_id:02d}"
            frames_dir = self.data_root / vid_name / "frames"
            phase_file = self.data_root / vid_name / "phase_annotations.txt"
            tool_file = self.data_root / vid_name / "tool_annotations.txt"
            
            if not frames_dir.exists():
                continue
            
            # Load phase annotations
            phase_annotations = {}
            if phase_file.exists():
                with open(phase_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            frame_idx = int(parts[0])
                            phase = int(parts[1])
                            phase_annotations[frame_idx] = phase
            
            # Load tool annotations
            tool_annotations = {}
            if self.include_tools and tool_file.exists():
                with open(tool_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:  # Skip header
                        parts = line.strip().split('\t')
                        if len(parts) >= 8:
                            frame_idx = int(parts[0])
                            tools = [int(x) for x in parts[1:8]]
                            tool_annotations[frame_idx] = tools
            
            # Collect frames — names like "frame_00000025.jpg" encode the
            # original 25fps frame index. Files on disk are already sampled
            # by prepare_data.py, so use all of them and parse the index
            # from the filename rather than re-sampling here.
            frame_files = sorted(frames_dir.glob("frame_*.jpg"))
            if not frame_files:
                frame_files = sorted(frames_dir.glob("*.png")) or sorted(frames_dir.glob("*.jpg"))

            for frame_path in frame_files:
                try:
                    frame_idx = int(frame_path.stem.replace("frame_", ""))
                except ValueError:
                    continue

                # Get phase label
                phase = phase_annotations.get(frame_idx, -1)
                if phase == -1:
                    closest_key = min(phase_annotations.keys(),
                                      key=lambda x: abs(x - frame_idx),
                                      default=None)
                    if closest_key is None:
                        continue
                    phase = phase_annotations[closest_key]

                # Get tool labels (tool ann is at 1fps, fall back to nearest)
                if self.include_tools and tool_annotations:
                    tools = tool_annotations.get(frame_idx)
                    if tools is None:
                        closest_tool_key = min(tool_annotations.keys(),
                                                key=lambda x: abs(x - frame_idx))
                        tools = tool_annotations[closest_tool_key]
                else:
                    tools = [0] * NUM_TOOLS

                self.frames.append({
                    'path': str(frame_path),
                    'phase': phase,
                    'tools': tools,
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                })
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame_info = self.frames[idx]
        
        # Load image
        image = Image.open(frame_info['path']).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        phase = torch.tensor(frame_info['phase'], dtype=torch.long)
        tools = torch.tensor(frame_info['tools'], dtype=torch.float32)
        
        return {
            'image': image,
            'phase': phase,
            'tools': tools,
            'video_id': frame_info['video_id'],
            'frame_idx': frame_info['frame_idx'],
        }


class Cholec80SequenceDataset(Dataset):
    """
    Cholec80 Sequence-Level Dataset.
    
    Loads temporal sequences of frames for training models
    with temporal reasoning (LSTM, TCN, Transformer).
    
    Args:
        data_root: Root directory containing video frames and annotations
        video_ids: List of video IDs to include (1-80)
        transform: Image transforms to apply
        sequence_length: Number of frames per sequence
        stride: Stride between consecutive sequences
        fps: Frames per second to sample
        include_tools: Whether to load tool annotations
        return_all_phases: If True, return phase for each frame;
                         if False, return phase for last frame only
    """
    
    def __init__(
        self,
        data_root: str,
        video_ids: List[int],
        transform=None,
        sequence_length: int = 16,
        stride: int = 8,
        fps: int = 1,
        include_tools: bool = True,
        return_all_phases: bool = True,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.video_ids = video_ids
        self.transform = transform
        self.sequence_length = sequence_length
        self.stride = stride
        self.fps = fps
        self.include_tools = include_tools
        self.return_all_phases = return_all_phases
        
        # First load all frames per video
        self.video_frames = {}  # vid_id -> list of frame dicts
        self.sequences = []  # list of (vid_id, start_idx)
        
        self._load_all_frames()
        self._create_sequences()
    
    def _load_all_frames(self):
        """Load all frames metadata organized by video."""
        frame_dataset = Cholec80Dataset(
            data_root=self.data_root,
            video_ids=self.video_ids,
            transform=None,  # Apply transform in __getitem__
            fps=self.fps,
            include_tools=self.include_tools,
        )
        
        for frame_info in frame_dataset.frames:
            vid_id = frame_info['video_id']
            if vid_id not in self.video_frames:
                self.video_frames[vid_id] = []
            self.video_frames[vid_id].append(frame_info)
    
    def _create_sequences(self):
        """Create sliding window sequences from each video."""
        for vid_id, frames in self.video_frames.items():
            num_frames = len(frames)
            if num_frames < self.sequence_length:
                # Pad short videos
                self.sequences.append((vid_id, 0))
                continue
            
            for start in range(0, num_frames - self.sequence_length + 1, self.stride):
                self.sequences.append((vid_id, start))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        vid_id, start_idx = self.sequences[idx]
        frames = self.video_frames[vid_id]
        
        # Get sequence frames
        end_idx = start_idx + self.sequence_length
        seq_frames = frames[start_idx:end_idx]
        
        # Handle padding for short sequences
        while len(seq_frames) < self.sequence_length:
            seq_frames.append(seq_frames[-1])
        
        # Load images and labels
        images = []
        phases = []
        tools_list = []
        
        for frame_info in seq_frames:
            image = Image.open(frame_info['path']).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
            phases.append(frame_info['phase'])
            tools_list.append(frame_info['tools'])
        
        images = torch.stack(images)  # (T, C, H, W)
        phases = torch.tensor(phases, dtype=torch.long)  # (T,)
        tools = torch.tensor(tools_list, dtype=torch.float32)  # (T, 7)
        
        if not self.return_all_phases:
            phases = phases[-1]  # Only last frame phase
            tools = tools[-1]
        
        return {
            'images': images,
            'phases': phases,
            'tools': tools,
            'video_id': vid_id,
            'start_idx': start_idx,
        }


class SyntheticCholec80Dataset(Dataset):
    """
    Synthetic dataset for development and testing.
    
    Generates random frames with consistent phase patterns
    to allow model development without the actual Cholec80 data.
    
    Args:
        num_videos: Number of synthetic videos to generate
        frames_per_video: Number of frames per video
        sequence_length: Temporal window size
        transform: Image transforms
        img_size: Size of generated images
    """
    
    def __init__(
        self,
        num_videos: int = 10,
        frames_per_video: int = 200,
        sequence_length: int = 16,
        stride: int = 8,
        transform=None,
        img_size: int = 224,
        include_tools: bool = True,
    ):
        super().__init__()
        self.num_videos = num_videos
        self.frames_per_video = frames_per_video
        self.sequence_length = sequence_length
        self.stride = stride
        self.transform = transform
        self.img_size = img_size
        self.include_tools = include_tools
        
        # Generate synthetic phase sequences
        self.video_data = []
        self.sequences = []
        self._generate_data()
    
    def _generate_data(self):
        """Generate synthetic video data with realistic phase transitions."""
        # Phase transition probabilities (simplified surgical workflow)
        # Phases tend to occur in order: 0->1->2->3->4->5->6
        phase_colors = {
            0: (200, 100, 100),   # Preparation - reddish
            1: (100, 200, 100),   # CalotTriangleDissection - greenish
            2: (100, 100, 200),   # ClippingCutting - bluish
            3: (200, 200, 100),   # GallbladderDissection - yellowish
            4: (200, 100, 200),   # GallbladderPackaging - magenta
            5: (100, 200, 200),   # CleaningCoagulation - cyan
            6: (150, 150, 150),   # GallbladderRetraction - grey
        }
        
        # Tool-phase associations (which tools are likely in each phase)
        tool_phase_probs = {
            0: [0.8, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],  # Preparation
            1: [0.9, 0.3, 0.8, 0.1, 0.0, 0.1, 0.0],  # Calot
            2: [0.7, 0.2, 0.4, 0.3, 0.9, 0.0, 0.0],  # Clipping
            3: [0.8, 0.3, 0.9, 0.2, 0.0, 0.2, 0.0],  # Dissection
            4: [0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.8],  # Packaging
            5: [0.3, 0.1, 0.2, 0.0, 0.0, 0.9, 0.0],  # Cleaning
            6: [0.9, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0],  # Retraction
        }
        
        for vid_id in range(self.num_videos):
            frames = []
            
            # Generate phase sequence (phases roughly in order)
            phases = self._generate_phase_sequence(self.frames_per_video)
            
            for frame_idx, phase in enumerate(phases):
                # Generate tool labels based on phase
                tools = [1 if random.random() < p else 0 
                        for p in tool_phase_probs[phase]]
                
                frames.append({
                    'phase': phase,
                    'tools': tools,
                    'video_id': vid_id,
                    'frame_idx': frame_idx,
                    'color': phase_colors[phase],
                })
            
            self.video_data.append(frames)
            
            # Create sequences
            for start in range(0, len(frames) - self.sequence_length + 1, self.stride):
                self.sequences.append((vid_id, start))
    
    def _generate_phase_sequence(self, length: int) -> List[int]:
        """Generate a realistic phase sequence."""
        phases = []
        current_phase = 0
        phase_duration = random.randint(15, 40)
        duration_counter = 0
        
        for _ in range(length):
            phases.append(current_phase)
            duration_counter += 1
            
            if duration_counter >= phase_duration:
                if current_phase < NUM_PHASES - 1:
                    current_phase += 1
                    phase_duration = random.randint(15, 40)
                    duration_counter = 0
        
        return phases
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        vid_id, start_idx = self.sequences[idx]
        frames = self.video_data[vid_id]
        
        seq_frames = frames[start_idx:start_idx + self.sequence_length]
        
        # Pad if necessary
        while len(seq_frames) < self.sequence_length:
            seq_frames.append(seq_frames[-1])
        
        images = []
        phases = []
        tools_list = []
        
        for frame_info in seq_frames:
            # Generate synthetic image with phase-specific color
            img = self._generate_frame(frame_info)
            if self.transform:
                img = self.transform(img)
            else:
                img = torch.tensor(np.transpose(np.array(img), (2, 0, 1)), 
                                 dtype=torch.float32) / 255.0
            images.append(img)
            phases.append(frame_info['phase'])
            tools_list.append(frame_info['tools'])
        
        images = torch.stack(images)
        phases = torch.tensor(phases, dtype=torch.long)
        tools = torch.tensor(tools_list, dtype=torch.float32)
        
        return {
            'images': images,
            'phases': phases,
            'tools': tools,
            'video_id': vid_id,
            'start_idx': start_idx,
        }
    
    def _generate_frame(self, frame_info: dict) -> Image.Image:
        """Generate a synthetic frame with phase-specific visual features."""
        color = np.array(frame_info['color'], dtype=np.int16)
        s = self.img_size
        
        # Vectorized: base color + uniform noise
        noise = np.random.randint(-30, 30, (s, s, 3), dtype=np.int16)
        img = np.clip(color + noise, 0, 255).astype(np.uint8)
        
        # Add circles (tools/tissue)
        for _ in range(random.randint(2, 5)):
            cx, cy = random.randint(30, s - 30), random.randint(30, s - 30)
            r = random.randint(10, 30)
            sc = tuple(np.random.randint(50, 255, 3).tolist())
            cv2.circle(img, (cx, cy), r, sc, -1)
        
        # Add tool-like lines
        for present in frame_info['tools']:
            if present:
                p1 = (random.randint(0, s), random.randint(0, s))
                p2 = (random.randint(0, s), random.randint(0, s))
                cv2.line(img, p1, p2, (200, 200, 200), 2)
        
        return Image.fromarray(img)

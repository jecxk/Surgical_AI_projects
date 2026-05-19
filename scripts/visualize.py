"""
Visualization Script for Surgical Phase Recognition.

Generates Grad-CAM, timelines, and analysis plots.

Usage:
    python scripts/visualize.py --checkpoint results/resnet50_lstm/checkpoints/best_model.pth
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset.cholec80_dataset import SyntheticCholec80Dataset
from src.dataset.transforms import get_val_transforms
from src.models.surgical_model import SurgicalPhaseModel
from src.visualization.gradcam import SurgicalGradCAM
from src.visualization.temporal_plot import plot_phase_timeline, plot_confusion_matrix
from src.visualization.tool_phase_analysis import plot_tool_phase_correlation


def main():
    parser = argparse.ArgumentParser(description="Visualize Model Results")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default='results/visualizations')
    parser.add_argument('--num-gradcam', type=int, default=10)
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']
    model_config = config if 'backbone' in config else config.get('model', config)
    model_config['num_phases'] = 7
    model_config['num_tools'] = 7
    
    model = SurgicalPhaseModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create synthetic data for visualization
    val_transform = get_val_transforms(224)
    dataset = SyntheticCholec80Dataset(
        num_videos=3, frames_per_video=200,
        sequence_length=16, stride=16,
        transform=val_transform,
    )
    
    class_names = [
        "Preparation", "CalotTriangleDissection", "ClippingCutting",
        "GallbladderDissection", "GallbladderPackaging",
        "CleaningCoagulation", "GallbladderRetraction",
    ]
    
    # 1. Generate Grad-CAM visualizations
    print("Generating Grad-CAM visualizations...")
    gradcam_dir = output_dir / 'gradcam'
    gradcam_dir.mkdir(exist_ok=True)
    
    if model.backbone is not None:
        try:
            target_layer = model.backbone.get_target_layer()
            gradcam = SurgicalGradCAM(model, target_layer, device)
            
            for i in range(min(args.num_gradcam, len(dataset))):
                sample = dataset[i]
                images = sample['images']  # (T, C, H, W)
                
                # Use middle frame
                mid = images.shape[0] // 2
                image = images[mid]
                
                # Create a simple original image for overlay
                orig = np.random.randint(100, 200, (224, 224, 3), dtype=np.uint8)
                
                gradcam.visualize(
                    image, orig,
                    class_names=class_names,
                    save_path=str(gradcam_dir / f'gradcam_sample_{i}.png'),
                )
            print(f"  Saved {min(args.num_gradcam, len(dataset))} Grad-CAM images")
        except Exception as e:
            print(f"  Grad-CAM failed: {e}")
    
    # 2. Generate phase timeline
    print("Generating phase timelines...")
    timeline_dir = output_dir / 'timelines'
    timeline_dir.mkdir(exist_ok=True)
    
    all_preds = []
    all_targets = []
    all_tools = []
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    
    with torch.no_grad():
        for batch in loader:
            images = batch['images'].to(device)
            outputs = model(images)
            preds = outputs['phase_logits'].argmax(dim=-1).cpu()
            all_preds.append(preds.flatten())
            all_targets.append(batch['phases'].flatten())
            all_tools.append(batch['tools'].flatten(0, 1))
    
    all_preds = torch.cat(all_preds).numpy()
    all_targets = torch.cat(all_targets).numpy()
    all_tools = torch.cat(all_tools).numpy()
    
    plot_phase_timeline(
        all_targets[:500], all_preds[:500],
        title="Surgical Phase Timeline (Sample)",
        save_path=str(timeline_dir / 'phase_timeline.png'),
    )
    
    # 3. Confusion matrix
    print("Generating confusion matrix...")
    from src.evaluation.metrics import compute_confusion_matrix
    cm = compute_confusion_matrix(all_targets, all_preds)
    plot_confusion_matrix(
        cm, save_path=str(output_dir / 'confusion_matrix.png'),
    )
    
    # 4. Tool-phase correlation
    print("Generating tool-phase correlation...")
    plot_tool_phase_correlation(
        all_targets, all_tools,
        save_path=str(output_dir / 'tool_phase_correlation.png'),
    )
    
    print(f"\n✅ All visualizations saved to: {output_dir}")


if __name__ == '__main__':
    main()

import os
import sys
import json
import base64
import torch
import cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_file
import tempfile

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.surgical_model import SurgicalPhaseModel
from src.dataset.transforms import get_val_transforms

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Mock config for demo purposes when model isn't fully trained
DEMO_MODE = True
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PHASE_NAMES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction"
]

TOOL_NAMES = [
    "Grasper", "Bipolar", "Hook", "Scissors", "Clipper", "Irrigator", "SpecimenBag"
]

def load_demo_model():
    global model, DEMO_MODE
    model_path = Path('results/resnet50_lstm/checkpoints/best_model.pth')
    if model_path.exists():
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            config = checkpoint['config']
            model_config = config.get('model', config)
            model_config['num_phases'] = 7
            model_config['num_tools'] = 7
            
            model = SurgicalPhaseModel(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model = model.to(device)
            model.eval()
            DEMO_MODE = False
            print("Loaded real model!")
        except Exception as e:
            print(f"Failed to load model, using demo mode: {e}")
    else:
        print("Model not found, using demo mode")

load_demo_model()

@app.route('/')
def index():
    return render_template('index.html', phases=PHASE_NAMES, tools=TOOL_NAMES)

@app.route('/api/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video provided'}), 400
        
    video = request.files['video']
    if video.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_video.mp4')
    video.save(filepath)
    
    # Extract some frames and mock/run predictions
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample 20 frames for the demo
    num_samples = 20
    interval = max(1, total_frames // num_samples)
    
    results = []
    
    transform = get_val_transforms(224)
    
    current_phase = 0
    for i in range(num_samples):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if DEMO_MODE:
            # Generate realistic-looking fake predictions
            if i > 0 and np.random.random() < 0.15:
                current_phase = min(6, current_phase + 1)
            
            phase = current_phase
            confidence = np.random.uniform(0.7, 0.99)
            
            tools = []
            if phase in [1, 2, 3]:
                tools = ["Grasper", "Hook"] if np.random.random() > 0.5 else ["Grasper", "Scissors"]
            elif phase == 5:
                tools = ["Irrigator", "Bipolar"]
        else:
            # Run actual model
            img_tensor = transform(frame_rgb).unsqueeze(0).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(img_tensor)
                probs = torch.softmax(out['phase_logits'][0, 0], dim=0)
                phase = probs.argmax().item()
                confidence = probs[phase].item()
                
                tool_probs = torch.sigmoid(out['tool_logits'][0, 0])
                tools = [TOOL_NAMES[j] for j, p in enumerate(tool_probs) if p > 0.5]
                
        # Base64 encode the frame for frontend display
        _, buffer = cv2.imencode('.jpg', cv2.resize(frame, (320, 180)))
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        results.append({
            'timestamp': (i * interval) / fps,
            'phase': PHASE_NAMES[phase],
            'phase_id': phase,
            'confidence': float(confidence),
            'tools': tools,
            'frame': f"data:image/jpeg;base64,{frame_b64}"
        })
        
    cap.release()
    
    return jsonify({
        'success': True,
        'results': results,
        'video_info': {
            'duration': total_frames / fps,
            'fps': fps,
            'total_frames': total_frames
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)

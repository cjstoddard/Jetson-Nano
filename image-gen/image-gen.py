import os
import io
import base64
from flask import Flask, request, render_template_string, jsonify, send_file
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from datetime import datetime

# ------------------ Config ------------------
MODEL_ID = os.getenv("MODEL_ID", "stabilityai/stable-diffusion-2-1-base")

# Check GPU availability
print("=" * 60)
print("Checking GPU availability...")
print("=" * 60)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    DEVICE = "cuda"
else:
    print("WARNING: CUDA not available, using CPU (will be very slow!)")
    DEVICE = "cpu"
print("=" * 60)

OUTPUT_DIR = "/app/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)


# ------------------ Load Model ------------------
print("=" * 60)
print("Loading Stable Diffusion Model")
print("=" * 60)
print(f"Model: {MODEL_ID}")
print(f"Device: {DEVICE}")
print("This will take 2-5 minutes on first run...")
print("=" * 60)

pipe = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
    safety_checker=None,  # Disable for speed
    requires_safety_checker=False
)

# Use DPM-Solver for faster generation
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to(DEVICE)

# Enable memory optimizations for Jetson
if DEVICE == "cuda":
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

print("\n✓ Model loaded and ready!\n")


# ------------------ HTML Template ------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>AI Image Generator</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            text-align: center;
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        .subtitle { color: #666; }
        .main-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .panel {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        .panel h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 20px;
        }
        textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e8ed;
            border-radius: 12px;
            font-family: inherit;
            font-size: 14px;
            resize: vertical;
            min-height: 120px;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .controls {
            margin: 20px 0;
        }
        .control-group {
            margin-bottom: 15px;
        }
        label {
            display: block;
            margin-bottom: 5px;
            color: #666;
            font-weight: 600;
            font-size: 13px;
        }
        input[type="number"], select {
            width: 100%;
            padding: 10px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 14px;
        }
        input[type="number"]:focus, select:focus {
            outline: none;
            border-color: #667eea;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:active { transform: translateY(0); }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        #imageDisplay {
            width: 100%;
            min-height: 400px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #f8f9fa;
            border-radius: 12px;
            position: relative;
            overflow: hidden;
        }
        #imageDisplay img {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
        }
        .placeholder {
            color: #999;
            text-align: center;
            padding: 40px;
        }
        .status {
            margin-top: 15px;
            padding: 12px;
            border-radius: 8px;
            display: none;
        }
        .status.active { display: block; }
        .status.generating {
            background: #e3f2fd;
            color: #1976d2;
        }
        .status.success {
            background: #e8f5e9;
            color: #388e3c;
        }
        .status.error {
            background: #ffebee;
            color: #d32f2f;
        }
        .progress {
            width: 100%;
            height: 4px;
            background: #e1e8ed;
            border-radius: 2px;
            margin-top: 10px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #667eea, #764ba2);
            width: 0%;
            transition: width 0.3s;
            animation: pulse 1.5s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .tips {
            background: #fff3cd;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #ffc107;
            margin-top: 15px;
        }
        .tips h3 {
            color: #856404;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .tips ul {
            margin-left: 20px;
            color: #856404;
            font-size: 13px;
        }
        @media (max-width: 768px) {
            .main-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎨 AI Image Generator</h1>
            <p class="subtitle">Powered by Stable Diffusion on Jetson</p>
        </div>
        
        <div class="main-grid">
            <div class="panel">
                <h2>Generate Image</h2>
                
                <label>Prompt (describe what you want)</label>
                <textarea 
                    id="prompt" 
                    placeholder="A beautiful sunset over mountains, highly detailed, 4k..."
                ></textarea>
                
                <div class="controls">
                    <div class="control-group">
                        <label>Negative Prompt (what to avoid)</label>
                        <textarea 
                            id="negative" 
                            placeholder="blurry, low quality, distorted..."
                            style="min-height: 60px;"
                        ></textarea>
                    </div>
                    
                    <div class="control-group">
                        <label>Steps (more = better quality, slower)</label>
                        <input type="number" id="steps" value="25" min="10" max="50">
                    </div>
                    
                    <div class="control-group">
                        <label>Guidance Scale (how closely to follow prompt)</label>
                        <input type="number" id="guidance" value="7.5" min="1" max="20" step="0.5">
                    </div>
                </div>
                
                <button id="generateBtn" onclick="generate()">
                    🎨 Generate Image
                </button>
                
                <div id="status" class="status"></div>
                
                <div class="tips">
                    <h3>💡 Tips for Better Results</h3>
                    <ul>
                        <li>Be specific and descriptive</li>
                        <li>Add style keywords: "photorealistic", "oil painting", "digital art"</li>
                        <li>Include quality terms: "highly detailed", "4k", "masterpiece"</li>
                        <li>First generation takes ~30-60 seconds on Jetson</li>
                        <li>Subsequent ones are faster (~20-40 seconds)</li>
                    </ul>
                </div>
            </div>
            
            <div class="panel">
                <h2>Generated Image</h2>
                <div id="imageDisplay">
                    <div class="placeholder">
                        <p>🖼️</p>
                        <p>Your generated image will appear here</p>
                    </div>
                </div>
                <button id="downloadBtn" onclick="downloadImage()" style="margin-top: 15px; display: none;">
                    💾 Download Image
                </button>
            </div>
        </div>
    </div>

    <script>
        let currentImageData = null;
        
        async function generate() {
            const prompt = document.getElementById('prompt').value.trim();
            if (!prompt) {
                alert('Please enter a prompt');
                return;
            }
            
            const button = document.getElementById('generateBtn');
            const status = document.getElementById('status');
            const imageDisplay = document.getElementById('imageDisplay');
            const downloadBtn = document.getElementById('downloadBtn');
            
            // Disable button
            button.disabled = true;
            button.textContent = '⏳ Generating...';
            
            // Show status
            status.className = 'status generating active';
            status.innerHTML = `
                Generating image... This may take 30-60 seconds
                <div class="progress"><div class="progress-bar" style="width: 100%;"></div></div>
            `;
            
            // Hide download button
            downloadBtn.style.display = 'none';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        negative_prompt: document.getElementById('negative').value,
                        num_inference_steps: parseInt(document.getElementById('steps').value),
                        guidance_scale: parseFloat(document.getElementById('guidance').value)
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    currentImageData = data.image;
                    imageDisplay.innerHTML = `<img src="data:image/png;base64,${data.image}" alt="Generated image">`;
                    downloadBtn.style.display = 'block';
                    
                    status.className = 'status success active';
                    status.textContent = `✓ Image generated successfully! (took ${data.time_taken}s)`;
                } else {
                    status.className = 'status error active';
                    status.textContent = `✗ Error: ${data.error}`;
                }
            } catch (err) {
                status.className = 'status error active';
                status.textContent = `✗ Error: ${err.message}`;
            } finally {
                button.disabled = false;
                button.textContent = '🎨 Generate Image';
            }
        }
        
        function downloadImage() {
            if (!currentImageData) return;
            
            const link = document.createElement('a');
            link.href = 'data:image/png;base64,' + currentImageData;
            link.download = `ai-generated-${Date.now()}.png`;
            link.click();
        }
        
        // Enter to generate
        document.getElementById('prompt').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                generate();
            }
        });
    </script>
</body>
</html>
"""


# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '').strip()
        negative_prompt = data.get('negative_prompt', 'blurry, low quality')
        steps = data.get('num_inference_steps', 25)
        guidance = data.get('guidance_scale', 7.5)
        
        if not prompt:
            return jsonify({'success': False, 'error': 'No prompt provided'})
        
        print(f"\n{'='*60}")
        print(f"Generating image...")
        print(f"Prompt: {prompt}")
        print(f"Steps: {steps}, Guidance: {guidance}")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        
        # Generate image
        with torch.inference_mode():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                height=512,
                width=512
            ).images[0]
        
        end_time = datetime.now()
        time_taken = (end_time - start_time).total_seconds()
        
        # Convert to base64
        img_io = io.BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)
        img_base64 = base64.b64encode(img_io.getvalue()).decode('utf-8')
        
        # Save to disk
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"generated_{timestamp}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        image.save(filepath)
        
        print(f"✓ Image generated in {time_taken:.1f}s")
        print(f"✓ Saved to: {filepath}\n")
        
        return jsonify({
            'success': True,
            'image': img_base64,
            'time_taken': f"{time_taken:.1f}",
            'filename': filename
        })
    
    except Exception as e:
        print(f"Error generating image: {e}")
        return jsonify({'success': False, 'error': str(e)})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting AI Image Generator on http://0.0.0.0:8081")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=8081, debug=False, threaded=False)

#!/usr/bin/env bash
# Build script for AI Image Generator with Stable Diffusion
# Optimized for NVIDIA Jetson

set -euo pipefail

echo "=========================================="
echo "AI Image Generator - Build Script"
echo "Stable Diffusion on Jetson"
echo "=========================================="
echo ""

# Check required files
echo "[*] Checking required files..."
if [ ! -f "image-gen.py" ]; then
    echo "ERROR: image-gen.py not found"
    exit 1
fi

if [ ! -f "docker-compose.yaml" ]; then
    echo "ERROR: docker-compose.yaml not found"
    exit 1
fi

echo "    ✓ All required files present"

# Clean start option
echo ""
read -p "Clean start? This will remove old containers (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[*] Stopping containers..."
    docker compose -f docker-compose.yaml down 2>/dev/null || true
    echo "    ✓ Clean slate ready"
fi

echo ""
echo "[*] Creating outputs directory..."
mkdir -p outputs
echo "    ✓ Directory created"

echo ""
echo "=========================================="
echo "Building Docker Container"
echo "=========================================="
echo ""
echo "This will:"
echo "  • Pull PyTorch base image (~8GB, one-time)"
echo "  • Install diffusers and dependencies"
echo "  • Set up Stable Diffusion"
echo ""
echo "This may take 10-20 minutes on first run..."
echo ""

docker compose -f docker-compose.yaml build

echo ""
echo "=========================================="
echo "Starting Image Generator"
echo "=========================================="
echo ""

docker compose -f docker-compose.yaml up -d

echo ""
echo "[*] Container starting up..."
echo "    Model will download on first run (~5GB)"
echo "    This takes 5-10 minutes depending on connection"
echo ""
echo "Waiting for service to be ready (checking every 10 seconds)..."

for i in {1..60}; do
    if curl -sf http://localhost:8081/ >/dev/null 2>&1; then
        echo ""
        echo "    ✓ Service is ready!"
        break
    fi
    
    if [ $i -eq 1 ]; then
        echo "    Downloading model and starting up..."
    elif [ $((i % 6)) -eq 0 ]; then
        echo "    Still loading... (${i}0 seconds elapsed)"
    fi
    
    printf "."
    sleep 10
    
    if [ $i -eq 60 ]; then
        echo ""
        echo "    ⚠ Service not responding after 10 minutes"
        echo "    Check logs: docker compose logs stable-diffusion"
        echo ""
        echo "Common causes:"
        echo "    • Model still downloading (check logs)"
        echo "    • Out of memory (check: sudo jtop)"
        echo "    • GPU not available"
        exit 1
    fi
done

echo ""
if docker compose -f docker-compose.yaml ps stable-diffusion | grep -q "Up"; then
    echo "=========================================="
    echo "✅ SUCCESS - Image Generator Running!"
    echo "=========================================="
    echo ""
    echo "🎨 Access your image generator:"
    echo "   http://localhost:8081"
    echo "   http://$(hostname -I | awk '{print $1}'):8081"
    echo ""
    echo "📊 Configuration:"
    echo "   • Model:  Stable Diffusion 2.1 Base"
    echo "   • Size:   512x512 images"
    echo "   • Speed:  ~30-60 seconds per image (first run)"
    echo "   •         ~20-40 seconds (subsequent)"
    echo ""
    echo "✨ Features:"
    echo "   ✓ Beautiful gradient UI"
    echo "   ✓ Adjustable quality settings"
    echo "   ✓ Negative prompts"
    echo "   ✓ Download generated images"
    echo "   ✓ Images saved to ./outputs/"
    echo ""
    echo "📝 Commands:"
    echo "   • View logs:  docker compose logs -f stable-diffusion"
    echo "   • Stop:       docker compose down"
    echo "   • Restart:    docker compose restart"
    echo ""
    echo "💡 Example prompts:"
    echo "   • 'A majestic lion in the savanna, golden hour, photorealistic'"
    echo "   • 'Cyberpunk city at night, neon lights, highly detailed'"
    echo "   • 'Oil painting of a mountain landscape, impressionist style'"
    echo ""
    echo "⚠️  Memory Usage:"
    echo "   • Expected: ~6-7GB GPU RAM"
    echo "   • Model on disk: ~5GB"
    echo "   • Check: sudo jtop"
    echo ""
else
    echo "=========================================="
    echo "⚠️ Container Status Unknown"
    echo "=========================================="
    echo ""
    echo "Check logs: docker compose logs stable-diffusion"
fi

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo ""
echo "Container status:"
docker compose -f docker-compose.yaml ps
echo ""
echo "Output directory:"
ls -lh outputs/ 2>/dev/null || echo "  (empty - no images generated yet)"
echo ""
echo "Disk usage:"
du -sh outputs/
echo ""
echo "=========================================="
echo "Build Complete! 🎨"
echo "=========================================="

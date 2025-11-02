#!/usr/bin/env bash
# Build script for ELIZA - Rogerian Therapist Chatbot
# Based on Carl Rogers' person-centered therapy

set -euo pipefail

echo "=========================================="
echo "ELIZA - Rogerian Therapist"
echo "Build Script"
echo "=========================================="
echo ""
echo "Grounded in Rogers' three core conditions:"
echo "  • Unconditional Positive Regard"
echo "  • Congruence (Genuineness)"
echo "  • Empathic Understanding"
echo ""

# Check required files
echo "[*] Checking required files..."
if [ ! -f "eliza.py" ]; then
    echo "ERROR: eliza.py not found"
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
    docker compose down 2>/dev/null || true
    echo "    ✓ Clean slate ready"
fi

echo ""
echo "[*] Creating directories..."
mkdir -p ollama
echo "    ✓ Directories created"

echo ""
echo "=========================================="
echo "Building Docker Containers"
echo "=========================================="
echo ""

docker compose build

echo ""
echo "=========================================="
echo "Starting Services"
echo "=========================================="
echo ""

# Start Ollama first
echo "[*] Starting Ollama..."
docker compose up -d ollama

echo ""
echo "[*] Waiting for Ollama to be ready..."
for i in {1..60}; do
    if docker compose exec -T ollama curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        echo "    ✓ Ollama is ready!"
        break
    fi
    printf "."
    sleep 1
    if [ $i -eq 60 ]; then
        echo ""
        echo "ERROR: Ollama failed to start"
        exit 1
    fi
done
echo ""

# Check if model exists
echo "[*] Checking for model (qwen2.5:3b)..."
if docker compose exec -T ollama ollama list | grep -q "qwen2.5:3b"; then
    echo "    ✓ Model already available"
else
    echo "    Model not found."
    echo ""
    echo "    Pulling qwen2.5:3b (~2GB, 3-5 minutes)..."
    echo "    This model is optimized for Jetson's memory constraints"
    echo "    while maintaining good emotional intelligence."
    echo ""
    docker compose exec -T ollama ollama pull qwen2.5:3b
    echo "    ✓ Model ready"
fi

# Start ELIZA
echo ""
echo "[*] Starting ELIZA..."
docker compose up -d eliza

echo ""
echo "[*] Waiting for ELIZA (15 seconds)..."
sleep 15

echo ""
if docker compose ps eliza | grep -q "Up"; then
    echo "=========================================="
    echo "✅ SUCCESS - ELIZA is Ready to Listen"
    echo "=========================================="
    echo ""
    echo "🌸 Access ELIZA:"
    echo "   http://localhost:8080"
    echo "   http://$(hostname -I | awk '{print $1}'):8080"
    echo ""
    echo "💭 Therapeutic Approach:"
    echo "   • Unconditional Positive Regard - Complete acceptance"
    echo "   • Congruence - Genuine, authentic responses"
    echo "   • Empathic Understanding - Deep reflective listening"
    echo "   • Facilitates actualizing tendency - Your natural growth"
    echo ""
    echo "🎯 Features:"
    echo "   ✓ Non-judgmental space"
    echo "   ✓ Reflective listening"
    echo "   ✓ Open-ended exploration"
    echo "   ✓ Emotion-focused dialogue"
    echo "   ✓ Client-centered approach"
    echo ""
    echo "⚠️  Important Reminders:"
    echo "   • For educational and self-exploration only"
    echo "   • NOT a replacement for professional therapy"
    echo "   • Crisis support: Call 988 (Suicide & Crisis Lifeline)"
    echo "   • Serious concerns: See a licensed professional"
    echo ""
    echo "💡 How to Use:"
    echo "   • Share your thoughts and feelings openly"
    echo "   • Take your time - no rush"
    echo "   • Notice how expressing yourself feels"
    echo "   • Trust your own capacity for growth"
    echo ""
    echo "📝 Commands:"
    echo "   • View logs:  docker compose logs -f eliza"
    echo "   • Stop:       docker compose down"
    echo "   • Restart:    docker compose restart"
    echo ""
    echo "📊 Configuration:"
    echo "   • Model:  qwen2.5:3b (balanced for Jetson)"
    echo "   • Port:   8081"
    echo "   • Memory: Conversation context maintained"
    echo ""
    echo "⚠️  Memory Usage:"
    echo "   • Expected: ~3-4GB GPU RAM"
    echo "   • Safe for Jetson Orin Nano"
    echo "   • Check: sudo jtop"
    echo ""
else
    echo "=========================================="
    echo "⚠️ ELIZA Status Unknown"
    echo "=========================================="
    echo ""
    echo "Check logs: docker compose logs eliza"
fi

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo ""
echo "Container status:"
docker compose ps
echo ""
echo "Available models:"
docker compose exec -T ollama ollama list 2>/dev/null || echo "  (could not retrieve)"
echo ""
echo "=========================================="
echo "Build Complete! 🌸"
echo "=========================================="
echo ""
echo "Remember: ELIZA is here to listen and support"
echo "your journey of self-exploration and growth."
echo ""

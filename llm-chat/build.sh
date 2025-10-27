#!/usr/bin/env bash
# Build script for Simple LLM Chat with Gemma-3-4B model
# Automatically downloads and imports GGUF model into Ollama
# Final working version with all fixes

set -euo pipefail

ROOT="$(pwd)"

echo "=========================================="
echo "Simple LLM Chat - Build Script"
echo "With Gemma-3-4B Model"
echo "=========================================="
echo ""

# Check required files
echo "[*] Checking required files..."
if [ ! -f "simple-chat.py" ]; then
    echo "ERROR: simple-chat.py not found in current directory"
    exit 1
fi

if [ ! -f "docker-compose.yaml" ]; then
    echo "ERROR: docker-compose.yaml not found in current directory"
    exit 1
fi

echo "    ✓ All required files present"

# Clean start option
echo ""
read -p "Clean start? This will stop containers and remove old models (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[*] Stopping containers..."
    docker compose -f docker-compose.yaml down 2>/dev/null || true
    
    echo "[*] Cleaning up old model files..."
    rm -f gemma-3-4b-it-abliterated-q5_k_m.gguf
    rm -f Modelfile
    rm -rf ollama
    
    echo "    ✓ Clean slate ready"
fi

echo ""
echo "[*] Creating ollama directory for persistent storage..."
mkdir -p ollama

echo ""
echo "[*] Building Docker containers..."
docker compose -f docker-compose.yaml build

echo ""
echo "[*] Starting Ollama service..."
docker compose -f docker-compose.yaml up -d ollama

echo ""
echo "[*] Waiting for Ollama to be ready (60 seconds max)..."
for i in {1..60}; do
  if docker compose -f docker-compose.yaml exec -T ollama curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "    ✓ Ollama is ready!"
    break
  fi
  printf "."
  sleep 1
  if [ $i -eq 60 ]; then
    echo ""
    echo "ERROR: Ollama failed to start after 60 seconds"
    echo "Check logs: docker compose logs ollama"
    exit 1
  fi
done
echo ""

echo ""
echo "=========================================="
echo "Downloading Gemma-3-4B Model"
echo "=========================================="
echo ""

MODEL_FILE="gemma-3-4b-it-abliterated-q5_k_m.gguf"
MODEL_URL="https://huggingface.co/ldostadi/gemma-3-4b-it-abliterated-Q5_K_M-GGUF/resolve/main/gemma-3-4b-it-abliterated-q5_k_m.gguf"

if [ -f "$MODEL_FILE" ]; then
    echo "[*] Model file already exists, skipping download"
    echo "    Delete $MODEL_FILE to re-download"
else
    echo "[*] Downloading Gemma-3-4B GGUF model (~3GB)..."
    echo "    This will take 5-15 minutes depending on connection"
    echo "    URL: $MODEL_URL"
    echo ""
    
    wget --progress=bar:force:noscroll "$MODEL_URL" -O "$MODEL_FILE"
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "ERROR: Download failed"
        echo "You can manually download from:"
        echo "$MODEL_URL"
        exit 1
    fi
    
    echo ""
    echo "    ✓ Download complete!"
fi

echo ""
echo "[*] Verifying file size..."
FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null)
FILE_SIZE_MB=$((FILE_SIZE / 1024 / 1024))
echo "    File size: ${FILE_SIZE_MB} MB"

if [ $FILE_SIZE_MB -lt 2500 ]; then
    echo "    WARNING: File seems too small (expected ~2800 MB)"
    echo "    Download may be incomplete"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "[*] Creating Ollama Modelfile with proper stop tokens..."
cat > Modelfile << 'EOF'
FROM ./gemma-3-4b-it-abliterated-q5_k_m.gguf

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 2048
PARAMETER stop "<|end|>"
PARAMETER stop "<|eot_id|>"
PARAMETER stop "</s>"
PARAMETER stop "<end>"
EOF

echo "    ✓ Modelfile created"

echo ""
echo "[*] Copying files to persistent volume..."
cp "$MODEL_FILE" ollama/
cp Modelfile ollama/

echo "    ✓ Files copied to ollama directory"

echo ""
echo "[*] Importing model into Ollama (this may take 2-5 minutes)..."
docker compose -f docker-compose.yaml exec -T ollama ollama create gemma-3-4b -f /root/.ollama/Modelfile

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to import model into Ollama"
    echo "Check logs: docker compose logs ollama"
    exit 1
fi

echo ""
echo "[*] Verifying model is loaded..."
docker compose -f docker-compose.yaml exec -T ollama ollama list

echo ""
echo "[*] Starting chat interface..."
docker compose -f docker-compose.yaml up -d chat

echo ""
echo "[*] Waiting for chat interface to start (10 seconds)..."
sleep 10

echo ""
if docker compose -f docker-compose.yaml ps chat | grep -q "Up"; then
  echo "=========================================="
  echo "✅ SUCCESS - Chat Interface Running!"
  echo "=========================================="
  echo ""
  echo "🎉 Access your chat:"
  echo "   http://localhost:8080"
  echo "   http://$(hostname -I | awk '{print $1}'):8080"
  echo ""
  echo "📊 Configuration:"
  echo "   • Model:     Gemma-3-4B (abliterated, Q5_K_M)"
  echo "   • Size:      2.8 GB"
  echo "   • Quality:   High (4 billion parameters)"
  echo "   • Context:   2048 tokens"
  echo "   • Uncensored: Yes (abliterated version)"
  echo "   • Persistent: Models saved in ./ollama/"
  echo ""
  echo "✨ Features:"
  echo "   ✓ Beautiful gradient UI"
  echo "   ✓ Conversation memory"
  echo "   ✓ Mobile responsive"
  echo "   ✓ Clear chat button"
  echo "   ✓ Models persist across restarts"
  echo ""
  echo "📝 Commands:"
  echo "   • View logs:  docker compose logs -f chat"
  echo "   • Stop:       docker compose down"
  echo "   • Restart:    docker compose restart"
  echo "   • Status:     docker compose ps"
  echo ""
  echo "💡 Test it:"
  echo "   1. Open http://localhost:8080 in browser"
  echo "   2. Type: 'What is pi?'"
  echo "   3. First response may take 10-20 seconds (model loading)"
  echo "   4. Enjoy intelligent conversation!"
  echo ""
  echo "🔍 Manual test (optional):"
  echo "   docker compose exec ollama ollama run gemma-3-4b 'hello'"
  echo ""
else
  echo "=========================================="
  echo "⚠️ Warning - Chat may have issues"
  echo "=========================================="
  echo ""
  echo "Check logs: docker compose logs chat"
  echo ""
  echo "Common issues:"
  echo "  1. Port in use → Check: netstat -tulpn | grep 8080"
  echo "  2. Model not loaded → Check: docker compose exec ollama ollama list"
  echo "  3. Out of memory → Check: sudo jtop"
  echo ""
fi

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo ""
echo "Files created:"
echo "  • $MODEL_FILE ($(du -h "$MODEL_FILE" | cut -f1))"
echo "  • Modelfile"
echo "  • ollama/ (persistent storage)"
echo ""
echo "Docker containers:"
docker compose -f docker-compose.yaml ps
echo ""
echo "Available models:"
docker compose -f docker-compose.yaml exec -T ollama ollama list
echo ""
echo "Disk usage:"
du -sh .
du -sh ollama
echo ""
echo "GPU Memory (check with: sudo jtop):"
echo "  • Expected usage: ~4-5 GB GPU RAM"
echo "  • Model file on disk: ~2.8 GB"
echo ""
echo "=========================================="
echo "Build Complete! 🚀"
echo "=========================================="
echo ""


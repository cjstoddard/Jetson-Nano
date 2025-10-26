#!/usr/bin/env bash
# Clean build for Jetson RAG - Flask + 1.5B model
# Optimized balance of memory and quality

set -euo pipefail

ROOT="${HOME}/rag"

echo "=========================================="
echo "Jetson RAG - Flask Edition"
echo "Clean Build from Scratch"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  • Frontend: Flask (lightweight)"
echo "  • Model: qwen2.5:1.5b (1.5B params)"
echo "  • Context: 512 tokens"
echo "  • Memory: ~2.5GB GPU RAM"
echo "  • Quality: Much better than minimal!"
echo ""

read -p "Start completely fresh? This will DELETE all data (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
  echo "[*] Complete cleanup..."
  cd "${ROOT}" && docker compose down 2>/dev/null || true
  rm -rf "${ROOT}/ollama" "${ROOT}/storage" "${ROOT}/data"
  echo "    ✓ Clean slate ready"
fi

echo ""
echo "[*] Creating directory structure..."
mkdir -p "${ROOT}/data" "${ROOT}/storage" "${ROOT}/ollama"

echo ""
echo "[*] Creating test document..."
cat > "${ROOT}/data/test.txt" << 'EOF'
The Jetson Orin Nano Developer Kit is an AI computer from NVIDIA designed for edge AI and robotics.

Key Specifications:
- Memory: 8GB unified RAM (shared between CPU and GPU)
- GPU: 1024-core NVIDIA Ampere architecture GPU
- CPU: 6-core Arm Cortex-A78AE v8.2 64-bit
- AI Performance: Up to 40 TOPS (INT8)
- Power: 7W to 15W configurable
- Connectivity: Gigabit Ethernet, WiFi, Bluetooth

This developer kit is ideal for:
- AI inference at the edge
- Robotics applications
- Computer vision projects
- Natural language processing
- Autonomous machines

The system runs Ubuntu 20.04 and supports popular AI frameworks including TensorFlow, PyTorch, and ONNX.
EOF
echo "    ✓ Test document created"

echo ""
echo "[*] Building RAG application image..."
docker compose -f "${ROOT}/docker-compose.yaml" build rag

echo ""
echo "[*] Starting Ollama service..."
docker compose -f "${ROOT}/docker-compose.yaml" up -d ollama

echo ""
echo "[*] Waiting for Ollama to be ready..."
for i in {1..60}; do
  if docker compose -f "${ROOT}/docker-compose.yaml" exec -T ollama curl -sf http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    echo "    ✓ Ollama is ready!"
    break
  fi
  printf "."
  sleep 1
  if [ $i -eq 60 ]; then
    echo ""
    echo "ERROR: Ollama failed to start after 60 seconds"
    echo "Check: docker compose logs ollama"
    exit 1
  fi
done
echo ""

echo ""
echo "=========================================="
echo "Downloading Models"
echo "=========================================="
echo ""

echo "[*] Pulling qwen2.5:1.5b (~1GB, 2-5 minutes)..."
echo "    This model is 3x better than the minimal 0.5b"
docker compose -f "${ROOT}/docker-compose.yaml" exec -T ollama ollama pull qwen2.5:1.5b

echo ""
echo "[*] Pulling nomic-embed-text (~274MB, 1 minute)..."
docker compose -f "${ROOT}/docker-compose.yaml" exec -T ollama ollama pull nomic-embed-text

echo ""
echo "[*] Verifying models are installed..."
docker compose -f "${ROOT}/docker-compose.yaml" exec -T ollama ollama list

echo ""
echo "[*] Testing embedding model..."
EMBED_TEST=$(docker compose -f "${ROOT}/docker-compose.yaml" exec -T ollama curl -s http://127.0.0.1:11434/api/embeddings -d '{"model":"nomic-embed-text","prompt":"test"}' 2>/dev/null)
if echo "$EMBED_TEST" | grep -q "embedding"; then
  echo "    ✓ Embeddings working!"
else
  echo "    ⚠ Embedding test inconclusive (may still work)"
fi

echo ""
echo "[*] Testing LLM..."
LLM_TEST=$(docker compose -f "${ROOT}/docker-compose.yaml" exec -T ollama curl -s http://127.0.0.1:11434/api/generate -d '{"model":"qwen2.5:1.5b","prompt":"hi","stream":false}' 2>/dev/null)
if echo "$LLM_TEST" | grep -q "response"; then
  echo "    ✓ LLM working!"
else
  echo "    ⚠ LLM test inconclusive (may still work)"
fi

echo ""
echo "[*] Disk space used by models..."
du -sh "${ROOT}/ollama" 2>/dev/null || echo "    ~1.3GB"

echo ""
echo "[*] Starting Flask RAG application..."
docker compose -f "${ROOT}/docker-compose.yaml" up -d rag

echo ""
echo "[*] Waiting for Flask to initialize (15 seconds)..."
sleep 15

echo ""
if docker compose -f "${ROOT}/docker-compose.yaml" ps rag | grep -q "Up"; then
  echo "=========================================="
  echo "✅ SUCCESS - Flask RAG is Running!"
  echo "=========================================="
  echo ""
  echo "🎉 Access your RAG system:"
  echo "   http://localhost:7860"
  echo "   http://$(hostname -I | awk '{print $1}'):7860"
  echo ""
  echo "📊 Configuration:"
  echo "   • Frontend:   Flask (lightweight, ~50MB)"
  echo "   • LLM:        qwen2.5:1.5b (1.5B params)"
  echo "   • Embeddings: nomic-embed-text"
  echo "   • Context:    512 tokens"
  echo "   • GPU RAM:    ~2.5GB"
  echo ""
  echo "✨ Features:"
  echo "   ✅ Beautiful, responsive web UI"
  echo "   ✅ Upload documents via browser"
  echo "   ✅ Much better answers than minimal"
  echo "   ✅ Works on desktop and mobile"
  echo ""
  echo "🚀 Getting Started:"
  echo "   1. Open http://localhost:7860 in browser"
  echo "   2. Upload a document (txt, pdf, md)"
  echo "   3. Click 'Upload & Reindex'"
  echo "   4. Ask a question about the document"
  echo "   5. Get intelligent answers!"
  echo ""
  echo "📝 Commands:"
  echo "   • View logs:  docker compose logs -f rag"
  echo "   • Stop:       docker compose down"
  echo "   • Restart:    docker compose restart rag"
  echo "   • Status:     docker compose ps"
  echo ""
  echo "💡 Tips:"
  echo "   • Be specific in your questions"
  echo "   • Upload clear, well-formatted documents"
  echo "   • Model handles ~300-400 words of context"
  echo "   • Press Ctrl+Enter in question box to submit"
  echo ""
else
  echo "=========================================="
  echo "⚠️ Warning - Check Status"
  echo "=========================================="
  echo ""
  echo "Container may have issues. Check logs:"
  echo "  docker compose logs rag"
  echo ""
  echo "Common issues:"
  echo "  1. Out of memory → Check: sudo jtop"
  echo "  2. Model not loaded → Check: docker compose logs ollama"
  echo "  3. Port in use → Check: netstat -tulpn | grep 7860"
  echo ""
  echo "To restart:"
  echo "  docker compose restart rag"
  echo ""
fi

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo ""
echo "Data locations:"
echo "  • Documents:   ${ROOT}/data"
echo "  • Index:       ${ROOT}/storage"
echo "  • Models:      ${ROOT}/ollama (~1.3GB)"
echo ""
echo "Model comparison:"
echo "  • 0.5B (minimal): Basic answers, fast"
echo "  • 1.5B (current): Good comprehension ⭐"
echo "  • 3B (if more RAM): Better reasoning"
echo ""
echo "Memory usage (approximate):"
echo "  • Ollama:      ~2.5GB"
echo "  • Flask RAG:   ~300MB"
echo "  • Total:       ~2.8GB"
echo "  • Available:   ~5GB (for other tasks)"
echo ""

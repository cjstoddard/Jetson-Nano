#!/usr/bin/env bash
# Build script for AI RAG Agent
# Inspired by n8n RAG setup from Hackster.io

set -euo pipefail

echo "=========================================="
echo "AI RAG Agent - Build Script"
echo "=========================================="
echo ""
echo "Based on: hackster.io/shahizat/build-local-ai-rag-agent-with-n8n"
echo "Simplified with Flask web interface instead of Telegram/n8n"
echo ""

# Check required files
echo "[*] Checking required files..."
if [ ! -f "rag-agent.py" ]; then
    echo "ERROR: rag-agent.py not found"
    exit 1
fi

if [ ! -f "docker-compose.yaml" ]; then
    echo "ERROR: docker-compose.yaml not found"
    exit 1
fi

echo "    ✓ All required files present"

# Clean start option
echo ""
read -p "Clean start? This will remove old data (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "[*] Cleaning old data..."
    docker compose down -v 2>/dev/null || true
    rm -rf ollama qdrant_storage uploads
    echo "    ✓ Clean slate ready"
fi

echo ""
echo "[*] Creating directories..."
mkdir -p ollama qdrant_storage uploads
echo "    ✓ Directories created"

echo ""
echo "=========================================="
echo "Building Docker Containers"
echo "=========================================="
echo ""
echo "This will:"
echo "  • Start Ollama for LLM and embeddings"
echo "  • Start Qdrant vector database"
echo "  • Build Flask RAG agent"
echo ""
echo "Build takes ~5 minutes..."
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

# Pull required models
echo "[*] Pulling AI models..."
echo "    This will take 5-15 minutes on first run"
echo ""

echo "    [1/2] Pulling LLM model (qwen2.5:3b)..."
docker compose exec -T ollama ollama pull qwen2.5:3b

echo ""
echo "    [2/2] Pulling embedding model (nomic-embed-text)..."
docker compose exec -T ollama ollama pull nomic-embed-text

echo ""
echo "    ✓ All models ready"

# Start Qdrant and RAG agent
echo ""
echo "[*] Starting Qdrant and RAG agent..."
docker compose up -d

echo ""
echo "[*] Waiting for RAG agent (30 seconds)..."
sleep 30

echo ""
if docker compose ps rag-agent | grep -q "Up"; then
    echo "=========================================="
    echo "✅ SUCCESS - RAG Agent Running!"
    echo "=========================================="
    echo ""
    echo "🌐 Access your RAG agent:"
    echo "   http://localhost:8080"
    echo "   http://$(hostname -I | awk '{print $1}'):8080"
    echo ""
    echo "📊 Component Ports:"
    echo "   • RAG Agent:  7861"
    echo "   • Ollama:     11437"
    echo "   • Qdrant:     6333 (API), 6334 (gRPC)"
    echo ""
    echo "🎯 Features:"
    echo "   ✓ Upload documents (TXT, PDF, HTML)"
    echo "   ✓ Ingest from URLs"
    echo "   ✓ Chat with your knowledge base"
    echo "   ✓ Conversational memory"
    echo "   ✓ Source attribution"
    echo ""
    echo "💡 How to use:"
    echo "   1. Upload documents or add URLs"
    echo "   2. Wait for processing"
    echo "   3. Ask questions about your documents"
    echo "   4. Get AI-powered answers with context!"
    echo ""
    echo "📝 Commands:"
    echo "   • View logs:     docker compose logs -f rag-agent"
    echo "   • View Ollama:   docker compose logs -f ollama"
    echo "   • View Qdrant:   docker compose logs -f qdrant"
    echo "   • Stop all:      docker compose down"
    echo "   • Restart:       docker compose restart"
    echo ""
    echo "🔧 Configuration:"
    echo "   • LLM Model:    qwen2.5:3b"
    echo "   • Embeddings:   nomic-embed-text"
    echo "   • Vector DB:    Qdrant"
    echo "   • Chunk size:   1000 tokens"
    echo ""
    echo "⚠️  Memory Usage:"
    echo "   • Expected: ~3-4GB GPU RAM"
    echo "   • Check: sudo jtop"
    echo ""
    echo "📖 Based on:"
    echo "   hackster.io/shahizat/build-local-ai-rag-agent-with-n8n"
    echo ""
else
    echo "=========================================="
    echo "⚠️ RAG Agent Status Unknown"
    echo "=========================================="
    echo ""
    echo "Check logs: docker compose logs rag-agent"
fi

echo ""
echo "=========================================="
echo "System Information"
echo "=========================================="
echo ""
echo "Container status:"
docker compose ps
echo ""
echo "Ollama models:"
docker compose exec -T ollama ollama list
echo ""
echo "Storage:"
du -sh ollama/ qdrant_storage/ uploads/
echo ""
echo "=========================================="
echo "Build Complete! 🤖"
echo "=========================================="

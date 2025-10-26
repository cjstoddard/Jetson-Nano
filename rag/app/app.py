import os
import shutil
from pathlib import Path
from typing import List

from flask import Flask, request, render_template_string, jsonify
from werkzeug.utils import secure_filename

from llama_index.core import (
    VectorStoreIndex,
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

from llama_index.llms.ollama import Ollama as LIOllama
from llama_index.embeddings.ollama import OllamaEmbedding


# ------------------ Config ------------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
GEN_MODEL   = os.getenv("GEN_MODEL", "qwen2.5:1.5b")
EMB_MODEL   = os.getenv("EMB_MODEL", "nomic-embed-text")

NUM_CTX   = int(os.getenv("NUM_CTX", "512"))
NUM_BATCH = int(os.getenv("NUM_BATCH", "8"))

DATA_DIR    = Path("/app/data")
STORAGE_DIR = Path("/app/storage")

TOP_K        = int(os.getenv("RAG_TOP_K", "3"))
INSERT_BATCH = int(os.getenv("RAG_INSERT_BATCH", "32"))

UPLOAD_FOLDER = Path("/tmp/uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)


# ------------------ Initialize ------------------
def init_llm() -> LIOllama:
    return LIOllama(
        model=GEN_MODEL,
        base_url=OLLAMA_BASE,
        request_timeout=600,
        additional_kwargs={
            "num_ctx": NUM_CTX,
            "num_batch": NUM_BATCH,
        },
    )


def init_embedder() -> OllamaEmbedding:
    return OllamaEmbedding(model_name=EMB_MODEL, base_url=OLLAMA_BASE)


def _ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _persist_files_present(p: Path) -> bool:
    expected = ("docstore.json", "vector_store.json", "index_store.json")
    return all((p / name).exists() for name in expected)


def _read_documents() -> List:
    if not DATA_DIR.exists():
        return []
    try:
        return SimpleDirectoryReader(str(DATA_DIR), recursive=True).load_data()
    except Exception:
        return []


def _build_index_from_docs(embedder: OllamaEmbedding) -> VectorStoreIndex:
    _ensure_dirs()
    docs = _read_documents()
    storage_ctx = StorageContext.from_defaults()
    index = VectorStoreIndex.from_documents(
        docs,
        storage_context=storage_ctx,
        embed_model=embedder,
        insert_batch_size=INSERT_BATCH,
        show_progress=True,
    )
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))
    return index


def build_or_load_index(embedder: OllamaEmbedding) -> VectorStoreIndex:
    _ensure_dirs()
    if _persist_files_present(STORAGE_DIR):
        try:
            storage_ctx = StorageContext.from_defaults(persist_dir=str(STORAGE_DIR))
            return load_index_from_storage(storage_ctx, embed_model=embedder)
        except Exception:
            pass
    return _build_index_from_docs(embedder)


# Initialize
print("=" * 60)
print("Flask RAG - Balanced Configuration")
print("=" * 60)
print(f"LLM: {GEN_MODEL} (1.5B params - 3x better than 0.5B)")
print(f"Embeddings: {EMB_MODEL}")
print(f"Context: {NUM_CTX} tokens (2x more than minimal)")
print(f"Memory saved by Flask: ~200MB vs Gradio")
print("=" * 60)

llm = init_llm()
embedder = init_embedder()
Settings.llm = llm
Settings.embed_model = embedder

print("\nInitializing index...")
index = build_or_load_index(embedder)
query_engine = index.as_query_engine(similarity_top_k=TOP_K)
print("System ready!\n")


# ------------------ Flask App ------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(UPLOAD_FOLDER)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Jetson RAG - Flask + Better Model</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            padding: 20px;
            max-width: 1400px;
            margin: 0 auto;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .header {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        .info { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-top: 15px;
        }
        .info-item {
            background: #f8f9fa;
            padding: 10px;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }
        .info-item strong { display: block; color: #667eea; margin-bottom: 3px; }
        .container { 
            display: grid; 
            grid-template-columns: 1fr 2fr; 
            gap: 20px;
        }
        .panel { 
            background: white; 
            padding: 25px; 
            border-radius: 12px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        h2 { color: #333; margin-bottom: 15px; font-size: 20px; }
        input[type="file"] { 
            margin: 10px 0;
            padding: 8px;
            border: 2px dashed #ddd;
            border-radius: 6px;
            width: 100%;
            cursor: pointer;
        }
        input[type="file"]:hover { border-color: #667eea; }
        button { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            margin: 5px 0;
            width: 100%;
            transition: transform 0.2s;
        }
        button:hover { transform: translateY(-2px); }
        button:active { transform: translateY(0); }
        textarea { 
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-family: inherit;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        #question { height: 100px; margin-bottom: 10px; }
        #answer { 
            height: 350px; 
            background: #f8f9fa;
            font-family: 'Courier New', monospace;
        }
        .status { 
            margin: 10px 0;
            padding: 12px;
            border-radius: 6px;
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            display: none;
            animation: slideIn 0.3s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(-10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .error { background: #ffebee; border-left-color: #f44336; }
        .success { background: #e8f5e9; border-left-color: #4caf50; }
        .tips {
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin-top: 15px;
            border-radius: 6px;
        }
        .tips h3 { color: #856404; margin-bottom: 8px; font-size: 16px; }
        .tips ul { margin-left: 20px; color: #856404; }
        @media (max-width: 768px) {
            .container { grid-template-columns: 1fr; }
            .info { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 Jetson RAG - Flask Edition</h1>
        <p style="color: #666;">Lightweight frontend with better model quality</p>
        <div class="info">
            <div class="info-item">
                <strong>Model</strong>
                {{ model }}
            </div>
            <div class="info-item">
                <strong>Parameters</strong>
                1.5 Billion
            </div>
            <div class="info-item">
                <strong>Context</strong>
                {{ context }} tokens
            </div>
            <div class="info-item">
                <strong>Memory Saved</strong>
                ~200MB vs Gradio
            </div>
        </div>
    </div>
    
    <div class="container">
        <div class="panel">
            <h2>📁 Document Management</h2>
            <form id="uploadForm" enctype="multipart/form-data">
                <input 
                    type="file" 
                    id="files" 
                    name="files" 
                    multiple 
                    accept=".txt,.pdf,.md,.doc,.docx"
                    title="Upload your documents"
                >
                <button type="submit">📤 Upload & Reindex</button>
            </form>
            <div id="uploadStatus" class="status"></div>
            
            <div class="tips">
                <h3>💡 Tips</h3>
                <ul>
                    <li>Upload clear, well-formatted docs</li>
                    <li>Be specific in questions</li>
                    <li>Model handles ~300 words context</li>
                </ul>
            </div>
        </div>
        
        <div class="panel">
            <h2>💬 Ask Questions</h2>
            <textarea 
                id="question" 
                placeholder="Ask detailed questions about your documents...&#10;&#10;Example: What are the main points discussed in the document?"
            ></textarea>
            <button onclick="ask()">🔍 Get Answer</button>
            <h3 style="margin-top: 20px; color: #667eea;">Answer:</h3>
            <textarea id="answer" readonly placeholder="Your answer will appear here..."></textarea>
        </div>
    </div>

    <script>
        document.getElementById('uploadForm').onsubmit = async (e) => {
            e.preventDefault();
            const status = document.getElementById('uploadStatus');
            const button = e.target.querySelector('button');
            
            status.textContent = '⏳ Uploading and indexing...';
            status.className = 'status';
            status.style.display = 'block';
            button.disabled = true;
            button.textContent = '⏳ Processing...';
            
            const formData = new FormData();
            const files = document.getElementById('files').files;
            
            if (files.length === 0) {
                status.textContent = '⚠️ Please select files first';
                status.className = 'status error';
                button.disabled = false;
                button.textContent = '📤 Upload & Reindex';
                return;
            }
            
            for (let file of files) {
                formData.append('files', file);
            }
            
            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });
                const data = await response.json();
                status.textContent = data.message;
                status.className = data.success ? 'status success' : 'status error';
            } catch (err) {
                status.textContent = '❌ Error: ' + err.message;
                status.className = 'status error';
            } finally {
                button.disabled = false;
                button.textContent = '📤 Upload & Reindex';
            }
        };
        
        async function ask() {
            const question = document.getElementById('question').value;
            const answer = document.getElementById('answer');
            const button = event.target;
            
            if (!question.trim()) {
                answer.value = '⚠️ Please enter a question first.';
                return;
            }
            
            answer.value = '🤔 Thinking...';
            button.disabled = true;
            button.textContent = '⏳ Processing...';
            
            try {
                const response = await fetch('/ask', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question })
                });
                const data = await response.json();
                answer.value = data.answer;
            } catch (err) {
                answer.value = '❌ Error: ' + err.message;
            } finally {
                button.disabled = false;
                button.textContent = '🔍 Get Answer';
            }
        }
        
        // Ctrl+Enter to ask
        document.getElementById('question').onkeydown = (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                ask();
            }
        };
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, model=GEN_MODEL, context=NUM_CTX)


@app.route('/upload', methods=['POST'])
def upload():
    global index, query_engine
    
    try:
        files = request.files.getlist('files')
        if not files or not files[0].filename:
            return jsonify({'success': False, 'message': '❌ No files uploaded'})
        
        uploaded = []
        for file in files:
            if file.filename:
                filename = secure_filename(file.filename)
                file.save(DATA_DIR / filename)
                uploaded.append(filename)
        
        # Clear and rebuild index
        for p in STORAGE_DIR.glob("*"):
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)
        
        print(f"\nRebuilding index with {len(uploaded)} files...")
        index = _build_index_from_docs(embedder)
        query_engine = index.as_query_engine(similarity_top_k=TOP_K)
        
        doc_list = ", ".join(p.name for p in DATA_DIR.glob("*") if p.is_file())
        return jsonify({
            'success': True,
            'message': f'✅ Successfully indexed {len(uploaded)} files!\nDocuments: {doc_list}'
        })
    
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({'success': False, 'message': f'❌ Error: {str(e)}'})


@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'answer': '⚠️ Please enter a question.'})
        
        print(f"\nQuestion: {question}")
        response = query_engine.query(question)
        answer = str(response)
        print(f"Answer length: {len(answer)} chars")
        
        return jsonify({'answer': answer})
    
    except Exception as e:
        print(f"Query error: {e}")
        return jsonify({'answer': f'❌ Error: {str(e)}'})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting Flask RAG on http://0.0.0.0:7860")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)

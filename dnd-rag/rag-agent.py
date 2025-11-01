import os
import io
import requests
from pathlib import Path
from datetime import datetime
from flask import Flask, request, render_template_string, jsonify, send_file
from werkzeug.utils import secure_filename

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
COLLECTION_NAME = "rag_documents"
UPLOAD_FOLDER = "/app/uploads"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

# Initialize components
print("=" * 60)
print("Initializing RAG Agent...")
print("=" * 60)
print(f"Ollama URL: {OLLAMA_BASE_URL}")
print(f"Qdrant URL: {QDRANT_URL}")
print(f"LLM Model: {LLM_MODEL}")
print(f"Embedding Model: {EMBED_MODEL}")
print("=" * 60)

# Initialize Qdrant
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize embeddings
embeddings = OllamaEmbeddings(
    model=EMBED_MODEL,
    base_url=OLLAMA_BASE_URL
)

# Initialize LLM
llm = Ollama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.7
)

# Create collection if it doesn't exist
try:
    qdrant_client.get_collection(COLLECTION_NAME)
    print(f"✓ Collection '{COLLECTION_NAME}' exists")
except:
    print(f"Creating collection '{COLLECTION_NAME}'...")
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    print(f"✓ Collection created")

# Initialize vector store
vectorstore = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)

# Conversation memory
memory = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=5  # Keep last 5 exchanges
)

# Custom prompt for D&D 5E Assistant
prompt_template = """You are an expert Dungeons & Dragons 5th Edition assistant and Dungeon Master helper. Your role is to help players and DMs with rules, mechanics, character creation, and gameplay questions using the D&D 5E SRD (System Reference Document) knowledge base.

Context from D&D 5E SRD:
{context}

Previous conversation:
{chat_history}

Player/DM Question: {question}

Instructions for your response:
1. **Rules Clarification**: When asked about rules, mechanics, or stats, provide clear, accurate information from the SRD context. Cite specific rules, spell details, or monster stats when relevant.

2. **Be Specific**: Include relevant numbers (AC, HP, damage dice, DC, etc.), action economy details, and mechanical specifics from the source material.

3. **Practical Application**: Explain not just WHAT the rule is, but HOW to use it in gameplay. Give examples when helpful.

4. **DM Guidance**: For DM questions, offer both RAW (Rules As Written) from the SRD and practical advice for table management.

5. **Character Building**: Help with character creation, optimization, and mechanical questions. Reference class features, spells, and abilities from the context.

6. **Combat & Actions**: Clearly explain action types (Action, Bonus Action, Reaction, Free Action), movement, and combat procedures.

7. **When Uncertain**: If the information isn't in your knowledge base, say "This specific rule isn't in the SRD I have access to" and suggest where to look (Player's Handbook, DMG, etc.).

8. **Stay in Character**: Maintain an enthusiastic, helpful tone befitting a knowledgeable DM. Use D&D terminology naturally.

9. **Format for Readability**: Use bullet points, numbered lists, or clear sections when explaining multiple mechanics or options.

Answer the question based on the SRD context provided above. Be precise, helpful, and game-ready:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "chat_history", "question"]
)

# Create RAG chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

print("\n✓ RAG Agent initialized and ready!\n")


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>D&D 5E AI Assistant</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            text-align: center;
        }
        h1 { 
            color: #667eea; 
            margin-bottom: 10px;
        }
        .subtitle { 
            color: #666;
        }
        .main-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
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
        .upload-area {
            border: 2px dashed #667eea;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .upload-area:hover { background: #f8f9fa; }
        .upload-area.dragging { background: #e3f2fd; border-color: #2196f3; }
        input[type="file"] { display: none; }
        .url-input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            margin-top: 15px;
        }
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
        }
        button:hover { 
            transform: translateY(-2px);
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            height: 600px;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 12px;
            margin-bottom: 20px;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            line-height: 1.5;
        }
        .message.user {
            background: #667eea;
            color: white;
            margin-left: auto;
        }
        .message.assistant {
            background: white;
            border: 1px solid #e1e8ed;
        }
        .message h1, .message h2, .message h3 {
            margin-top: 10px;
            margin-bottom: 10px;
            color: #667eea;
        }
        .message h1 { font-size: 20px; }
        .message h2 { font-size: 18px; }
        .message h3 { font-size: 16px; }
        .message ul, .message ol {
            margin: 10px 0;
            padding-left: 25px;
        }
        .message li {
            margin: 5px 0;
        }
        .message p {
            margin: 8px 0;
        }
        .message code {
            background: #f5f5f5;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Courier New', monospace;
            font-size: 13px;
        }
        .message pre {
            background: #f5f5f5;
            padding: 12px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }
        .message pre code {
            background: none;
            padding: 0;
        }
        .message strong {
            font-weight: 600;
            color: #333;
        }
        .message blockquote {
            border-left: 4px solid #667eea;
            padding-left: 15px;
            margin: 10px 0;
            color: #666;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
        }
        .input-area button {
            width: auto;
            margin: 0;
            padding: 12px 30px;
        }
        .status {
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
            display: none;
        }
        .status.active { display: block; }
        .status.success { 
            background: #e8f5e9; 
            color: #388e3c;
        }
        .status.error { 
            background: #ffebee; 
            color: #d32f2f;
        }
        .status.info { 
            background: #e3f2fd; 
            color: #1976d2;
        }
        .stats {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }
        .stats h3 { 
            font-size: 14px; 
            margin-bottom: 10px;
            color: #667eea;
        }
        .stats p { 
            font-size: 13px; 
            color: #666;
        }
        .tips {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border-left: 4px solid #2196f3;
        }
        .tips h3 {
            color: #1976d2;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .tips ul {
            margin-left: 20px;
            color: #1976d2;
            font-size: 13px;
        }
        .tips li {
            margin-bottom: 5px;
        }
        @media (max-width: 968px) {
            .main-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎲 D&D 5E AI Assistant 🐉</h1>
            <p class="subtitle">Your personal Dungeon Master's companion powered by the SRD</p>
        </div>
        
        <div class="main-grid">
            <div class="panel">
                <h2>📚 Knowledge Tome</h2>
                
                <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
                    <p>📜 Upload SRD Documents</p>
                    <p style="font-size: 12px; color: #999; margin-top: 10px;">
                        TXT, PDF, HTML files
                    </p>
                </div>
                <input type="file" id="fileInput" accept=".txt,.pdf,.html,.htm" multiple>
                
                <p style="margin: 20px 0; text-align: center; color: #999;">OR</p>
                
                <input type="url" id="urlInput" class="url-input" 
                       placeholder="https://www.5esrd.com/...">
                <button onclick="ingestUrl()">⚔️ Ingest from URL</button>
                
                <div id="uploadStatus" class="status"></div>
                
                <div class="stats">
                    <h3>📊 Grimoire Status</h3>
                    <p id="docCount">Scrolls indexed: Loading...</p>
                </div>
                
                <div class="tips">
                    <h3>💡 Suggested Sources</h3>
                    <ul>
                        <li><strong>5esrd.com</strong> - Complete SRD online</li>
                        <li><strong>Classes:</strong> Fighter, Wizard, Cleric, etc.</li>
                        <li><strong>Spells:</strong> Full spell descriptions</li>
                        <li><strong>Monsters:</strong> Stat blocks & tactics</li>
                        <li><strong>Rules:</strong> Combat, exploration, social</li>
                    </ul>
                </div>
            </div>
            
            <div class="panel">
                <h2>🗡️ Consult the Oracle</h2>
                
                <div class="chat-container">
                    <div class="messages" id="messages">
                        <div class="message assistant">
                            <p><strong>Greetings, adventurer!</strong> I am your D&D 5th Edition assistant. Upload the SRD or ingest from 5esrd.com, then ask me about:</p>
                            <ul>
                                <li>Rules & mechanics</li>
                                <li>Spell descriptions & effects</li>
                                <li>Monster stats & abilities</li>
                                <li>Character creation & leveling</li>
                                <li>Combat procedures</li>
                                <li>Conditions & status effects</li>
                            </ul>
                            <p><em>Example: "How does the Fireball spell work?" or "What are a Beholder's abilities?"</em></p>
                        </div>
                    </div>
                    
                    <div class="input-area">
                        <input type="text" id="messageInput" placeholder="Ask about rules, spells, monsters..." 
                               onkeypress="if(event.key=='Enter') sendMessage()">
                        <button onclick="sendMessage()">🎲 Ask</button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Markdown rendering library -->
    <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
    
    <script>
        // Configure marked for better formatting
        marked.setOptions({
            breaks: true,
            gfm: true
        });
        
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.add('dragging'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => uploadArea.classList.remove('dragging'), false);
        });
        
        uploadArea.addEventListener('drop', handleDrop, false);
        fileInput.addEventListener('change', handleFiles, false);
        
        function handleDrop(e) {
            const files = e.dataTransfer.files;
            handleFiles({ target: { files: files } });
        }
        
        async function handleFiles(e) {
            const files = Array.from(e.target.files);
            const status = document.getElementById('uploadStatus');
            
            status.className = 'status info active';
            status.textContent = `📖 Transcribing ${files.length} scroll(s)...`;
            
            for (const file of files) {
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        status.className = 'status success active';
                        status.textContent = `✓ ${file.name} added to the grimoire! (${data.chunks} chunks)`;
                    } else {
                        status.className = 'status error active';
                        status.textContent = `✗ Error: ${data.error}`;
                    }
                } catch (err) {
                    status.className = 'status error active';
                    status.textContent = `✗ Error uploading ${file.name}`;
                }
            }
            
            updateStats();
            fileInput.value = '';
        }
        
        async function ingestUrl() {
            const url = document.getElementById('urlInput').value.trim();
            const status = document.getElementById('uploadStatus');
            
            if (!url) {
                alert('Please enter a URL from 5esrd.com or another D&D resource');
                return;
            }
            
            status.className = 'status info active';
            status.textContent = '🔮 Channeling knowledge from the astral plane...';
            
            try {
                const response = await fetch('/ingest_url', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ url: url })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    status.className = 'status success active';
                    status.textContent = `✓ Knowledge absorbed! (${data.chunks} chunks from ${data.url})`;
                    document.getElementById('urlInput').value = '';
                } else {
                    status.className = 'status error active';
                    status.textContent = `✗ Error: ${data.error}`;
                }
            } catch (err) {
                status.className = 'status error active';
                status.textContent = `✗ Error: ${err.message}`;
            }
            
            updateStats();
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            const messagesDiv = document.getElementById('messages');
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.textContent = message;
            messagesDiv.appendChild(userMsg);
            
            input.value = '';
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Add thinking message
            const thinkingMsg = document.createElement('div');
            thinkingMsg.className = 'message assistant';
            thinkingMsg.textContent = '🎲 Consulting the tomes...';
            messagesDiv.appendChild(thinkingMsg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: message })
                });
                
                const data = await response.json();
                
                // Render markdown as HTML
                thinkingMsg.innerHTML = marked.parse(data.response);
            } catch (err) {
                thinkingMsg.textContent = `⚠️ Error: ${err.message}`;
                thinkingMsg.style.color = '#d32f2f';
            }
            
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        async function updateStats() {
            try {
                const response = await fetch('/stats');
                const data = await response.json();
                document.getElementById('docCount').textContent = 
                    `Scrolls indexed: ${data.count} knowledge chunks`;
            } catch (err) {
                console.error('Failed to update stats:', err);
            }
        }
        
        // Initial stats load
        updateStats();
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read file content
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Split and add to vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(content)
        
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"source": filename} for _ in chunks]
        )
        
        print(f"✓ Added {len(chunks)} chunks from {filename}")
        
        return jsonify({
            'success': True,
            'chunks': len(chunks),
            'filename': filename
        })
    
    except Exception as e:
        print(f"Error uploading file: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/ingest_url', methods=['POST'])
def ingest_url():
    try:
        data = request.get_json()
        url = data.get('url')
        
        if not url:
            return jsonify({'success': False, 'error': 'No URL provided'})
        
        # Fetch URL content
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Simple HTML stripping (you could use BeautifulSoup for better results)
        import re
        content = re.sub('<[^<]+?>', '', content)
        
        # Split and add to vector store
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(content)
        
        vectorstore.add_texts(
            texts=chunks,
            metadatas=[{"source": url} for _ in chunks]
        )
        
        print(f"✓ Added {len(chunks)} chunks from {url}")
        
        return jsonify({
            'success': True,
            'chunks': len(chunks),
            'url': url
        })
    
    except Exception as e:
        print(f"Error ingesting URL: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        
        if not message:
            return jsonify({'response': 'Please ask a question.'})
        
        print(f"\nUser: {message}")
        
        # Get response from RAG chain
        result = qa_chain({"question": message})
        response = result['answer']
        
        print(f"Assistant: {response}\n")
        
        return jsonify({'response': response})
    
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({'response': f'Error: {str(e)}'})


@app.route('/stats')
def stats():
    try:
        collection_info = qdrant_client.get_collection(COLLECTION_NAME)
        return jsonify({'count': collection_info.points_count})
    except Exception as e:
        return jsonify({'count': 0, 'error': str(e)})


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting AI RAG Agent on http://0.0.0.0:7861")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=7861, debug=False)

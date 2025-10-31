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
            background: linear-gradient(135deg, #8B0000 0%, #DC143C 50%, #B22222 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: linear-gradient(135deg, #2c1810 0%, #3d2415 100%);
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            margin-bottom: 20px;
            text-align: center;
            border: 3px solid #8B4513;
        }
        h1 { 
            color: #FFD700; 
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.8);
            font-size: 32px;
        }
        .subtitle { 
            color: #D4AF37;
            font-style: italic;
        }
        .main-grid { display: grid; grid-template-columns: 1fr 2fr; gap: 20px; }
        .panel {
            background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 100%);
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
            border: 2px solid #8B4513;
        }
        .panel h2 { 
            color: #8B0000; 
            margin-bottom: 20px; 
            font-size: 20px;
            border-bottom: 2px solid #8B4513;
            padding-bottom: 10px;
        }
        .upload-area {
            border: 2px dashed #8B0000;
            border-radius: 12px;
            padding: 30px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            background: rgba(139, 0, 0, 0.05);
        }
        .upload-area:hover { background: rgba(139, 0, 0, 0.1); }
        .upload-area.dragging { background: rgba(220, 20, 60, 0.2); border-color: #DC143C; }
        input[type="file"] { display: none; }
        .url-input {
            width: 100%;
            padding: 12px;
            border: 2px solid #8B4513;
            border-radius: 8px;
            margin-top: 15px;
            background: #fff;
        }
        button {
            background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%);
            color: #FFD700;
            border: 2px solid #8B4513;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            width: 100%;
            margin-top: 10px;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
        }
        button:hover { 
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
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
            background: linear-gradient(135deg, #2c1810 0%, #1a0f08 100%);
            border-radius: 12px;
            margin-bottom: 20px;
            border: 2px solid #8B4513;
        }
        .message {
            margin-bottom: 15px;
            padding: 12px 16px;
            border-radius: 12px;
            max-width: 80%;
            line-height: 1.5;
        }
        .message.user {
            background: linear-gradient(135deg, #8B0000 0%, #DC143C 100%);
            color: #FFD700;
            margin-left: auto;
            border: 1px solid #8B4513;
        }
        .message.assistant {
            background: linear-gradient(135deg, #f5f0e8 0%, #e8dcc8 100%);
            color: #2c1810;
            border: 1px solid #8B4513;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: 2px solid #8B4513;
            border-radius: 8px;
            background: #fff;
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
            border: 2px solid;
        }
        .status.active { display: block; }
        .status.success { 
            background: #90EE90; 
            color: #006400;
            border-color: #228B22;
        }
        .status.error { 
            background: #FFB6C1; 
            color: #8B0000;
            border-color: #DC143C;
        }
        .status.info { 
            background: #87CEEB; 
            color: #00008B;
            border-color: #4169E1;
        }
        .stats {
            background: rgba(139, 69, 19, 0.1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border: 1px solid #8B4513;
        }
        .stats h3 { 
            font-size: 14px; 
            margin-bottom: 10px;
            color: #8B0000;
        }
        .stats p { 
            font-size: 13px; 
            color: #2c1810;
        }
        .tips {
            background: rgba(255, 215, 0, 0.1);
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            border: 2px solid #FFD700;
        }
        .tips h3 {
            color: #8B0000;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .tips ul {
            margin-left: 20px;
            color: #2c1810;
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
                    <p style="font-size: 12px; color: #666; margin-top: 10px;">
                        TXT, PDF, HTML files
                    </p>
                </div>
                <input type="file" id="fileInput" accept=".txt,.pdf,.html,.htm" multiple>
                
                <p style="margin: 20px 0; text-align: center; color: #8B4513; font-weight: bold;">OR</p>
                
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
                            Greetings, adventurer! I am your D&D 5th Edition assistant. Upload the SRD or ingest from 5esrd.com, then ask me about:
                            <br><br>
                            • Rules & mechanics<br>
                            • Spell descriptions & effects<br>
                            • Monster stats & abilities<br>
                            • Character creation & leveling<br>
                            • Combat procedures<br>
                            • Conditions & status effects<br>
                            <br>
                            Example: "How does the Fireball spell work?" or "What are a Beholder's abilities?"
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

    <script>
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
                
                thinkingMsg.textContent = data.response;
            } catch (err) {
                thinkingMsg.textContent = `⚠️ Error: ${err.message}`;
                thinkingMsg.style.color = '#8B0000';
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

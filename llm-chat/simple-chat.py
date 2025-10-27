import os
from flask import Flask, request, render_template_string, jsonify, session
from datetime import datetime
import requests
import json

# ------------------ Config ------------------
OLLAMA_BASE = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:3b")

app = Flask(__name__)
app.secret_key = os.urandom(24)


# ------------------ HTML Template ------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat - {{ model }}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        .chat-container {
            background: white;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            width: 100%;
            max-width: 800px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        .chat-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }
        .chat-header h1 { font-size: 24px; margin-bottom: 5px; }
        .chat-header p { opacity: 0.9; font-size: 14px; }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f8f9fa;
        }
        .message {
            margin-bottom: 16px;
            display: flex;
            animation: slideIn 0.3s;
        }
        @keyframes slideIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.user { justify-content: flex-end; }
        .message.assistant { justify-content: flex-start; }
        .message-content {
            max-width: 70%;
            padding: 12px 16px;
            border-radius: 12px;
            word-wrap: break-word;
            white-space: pre-wrap;
        }
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
        }
        .message.assistant .message-content {
            background: white;
            color: #333;
            border: 1px solid #e1e8ed;
        }
        .timestamp {
            font-size: 11px;
            opacity: 0.6;
            margin-top: 4px;
        }
        .chat-input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e1e8ed;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        #messageInput {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e1e8ed;
            border-radius: 24px;
            font-size: 14px;
            font-family: inherit;
            transition: border-color 0.3s;
        }
        #messageInput:focus {
            outline: none;
            border-color: #667eea;
        }
        #sendButton {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 24px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            min-width: 80px;
        }
        #sendButton:hover { transform: scale(1.05); }
        #sendButton:active { transform: scale(0.95); }
        #sendButton:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .clear-button {
            background: #dc3545;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 16px;
            font-size: 12px;
            cursor: pointer;
            margin-top: 8px;
        }
        .clear-button:hover { background: #c82333; }
        .thinking {
            display: none;
            padding: 12px 16px;
            background: white;
            border: 1px solid #e1e8ed;
            border-radius: 12px;
            color: #666;
            max-width: 70%;
        }
        .thinking.active { display: block; }
        .thinking::after {
            content: '...';
            animation: dots 1.5s infinite;
        }
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        @media (max-width: 768px) {
            .chat-container { height: 100vh; border-radius: 0; }
            .message-content { max-width: 85%; }
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="chat-header">
            <h1>💬 AI Chat</h1>
            <p>Model: {{ model }}</p>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="message assistant">
                <div class="message-content">
                    Hello! I'm ready to chat. Ask me anything!
                    <div class="timestamp" id="initTime"></div>
                </div>
            </div>
        </div>
        
        <div class="chat-input-area">
            <div class="input-container">
                <input 
                    type="text" 
                    id="messageInput" 
                    placeholder="Type your message..."
                    autocomplete="off"
                >
                <button id="sendButton" onclick="sendMessage()">Send</button>
            </div>
            <button class="clear-button" onclick="clearChat()">🗑️ Clear Chat</button>
        </div>
    </div>

    <script>
        // Initialize timestamp
        document.getElementById('initTime').textContent = new Date().toLocaleTimeString();
        
        // Focus input on load
        document.getElementById('messageInput').focus();
        
        // Enter to send
        document.getElementById('messageInput').addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        function addMessage(content, isUser) {
            const messagesDiv = document.getElementById('messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'assistant'}`;
            
            const time = new Date().toLocaleTimeString();
            messageDiv.innerHTML = `
                <div class="message-content">
                    ${escapeHtml(content)}
                    <div class="timestamp">${time}</div>
                </div>
            `;
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function showThinking() {
            const messagesDiv = document.getElementById('messages');
            const thinkingDiv = document.createElement('div');
            thinkingDiv.className = 'message assistant';
            thinkingDiv.innerHTML = '<div class="thinking active">Thinking</div>';
            thinkingDiv.id = 'thinking';
            messagesDiv.appendChild(thinkingDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function hideThinking() {
            const thinking = document.getElementById('thinking');
            if (thinking) thinking.remove();
        }
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const button = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Disable input
            input.disabled = true;
            button.disabled = true;
            button.textContent = '...';
            
            // Add user message
            addMessage(message, true);
            input.value = '';
            
            // Show thinking
            showThinking();
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message })
                });
                
                const data = await response.json();
                
                hideThinking();
                
                if (data.response) {
                    addMessage(data.response, false);
                } else {
                    addMessage('Error: No response from model', false);
                }
            } catch (err) {
                hideThinking();
                addMessage('Error: ' + err.message, false);
            } finally {
                input.disabled = false;
                button.disabled = false;
                button.textContent = 'Send';
                input.focus();
            }
        }
        
        function clearChat() {
            if (confirm('Clear chat history?')) {
                fetch('/clear', { method: 'POST' });
                const messagesDiv = document.getElementById('messages');
                messagesDiv.innerHTML = `
                    <div class="message assistant">
                        <div class="message-content">
                            Chat cleared. How can I help you?
                            <div class="timestamp">${new Date().toLocaleTimeString()}</div>
                        </div>
                    </div>
                `;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
    </script>
</body>
</html>
"""


# ------------------ Routes ------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, model=MODEL_NAME)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Get or initialize conversation history
        if 'history' not in session:
            session['history'] = []
        
        # Add user message to history
        session['history'].append({
            'role': 'user',
            'content': user_message
        })
        
        # Call Ollama API
        response = requests.post(
            f"{OLLAMA_BASE}/api/chat",
            json={
                'model': MODEL_NAME,
                'messages': session['history'],
                'stream': False
            },
            timeout=120
        )
        
        if response.status_code != 200:
            return jsonify({'error': f'Ollama error: {response.text}'}), 500
        
        result = response.json()
        assistant_message = result['message']['content']
        
        # Add assistant response to history
        session['history'].append({
            'role': 'assistant',
            'content': assistant_message
        })
        
        # Keep only last 10 messages to avoid token overflow
        if len(session['history']) > 20:
            session['history'] = session['history'][-20:]
        
        session.modified = True
        
        return jsonify({'response': assistant_message})
    
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timed out'}), 504
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clear', methods=['POST'])
def clear():
    session['history'] = []
    session.modified = True
    return jsonify({'status': 'cleared'})


if __name__ == "__main__":
    print("=" * 60)
    print("Simple LLM Chat Interface")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Ollama: {OLLAMA_BASE}")
    print("Starting on http://0.0.0.0:8080")
    print("=" * 60)
    print()
    
    app.run(host="0.0.0.0", port=8080, debug=False, threaded=True)

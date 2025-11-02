import os
import requests
from flask import Flask, request, render_template_string, jsonify
from datetime import datetime

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "gemma-3-4b")

# ==============================================================================
# SYSTEM PROMPT - Edit this to change the assistant's behavior
# ==============================================================================
SYSTEM_PROMPT = """You are a helpful, friendly, and knowledgeable AI assistant. 
You provide clear, accurate, and concise responses while maintaining a conversational tone.

Key behaviors:
- Be helpful and informative
- Keep responses concise but complete
- Ask clarifying questions when needed
- Admit when you don't know something
- Be respectful and professional
- Format responses clearly with bullet points or numbered lists when appropriate

Remember: You are a general-purpose assistant here to help with a wide variety of tasks."""
# ==============================================================================

app = Flask(__name__)

print("=" * 60)
print("Initializing LLM Chat")
print("=" * 60)
print(f"Ollama URL: {OLLAMA_BASE_URL}")
print(f"Model: {MODEL}")
print("=" * 60)
print("✓ Chat is ready\n")


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Chat</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 900px; margin: 0 auto; }
        .header {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            margin-bottom: 20px;
            text-align: center;
        }
        h1 { color: #667eea; margin-bottom: 10px; }
        .subtitle { color: #666; font-style: italic; }
        .chat-panel {
            background: white;
            padding: 30px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
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
            line-height: 1.6;
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
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area input {
            flex: 1;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 15px;
        }
        .input-area input:focus {
            outline: none;
            border-color: #667eea;
        }
        .input-area button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 8px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
        }
        .input-area button:hover { transform: translateY(-2px); }
        .input-area button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
        }
        .controls button {
            background: #f8f9fa;
            color: #666;
            border: 1px solid #e1e8ed;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
        }
        .controls button:hover {
            background: #e9ecef;
            border-color: #667eea;
            color: #667eea;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>💬 LLM Chat</h1>
            <p class="subtitle">Powered by {{ MODEL }}</p>
        </div>
        
        <div class="chat-panel">
            <div class="controls">
                <button onclick="clearChat()">🔄 Clear Chat</button>
            </div>
            
            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="message assistant">
                        Hello! I'm your AI assistant. How can I help you today?
                    </div>
                </div>
                
                <div class="input-area">
                    <input 
                        type="text" 
                        id="messageInput" 
                        placeholder="Type your message..."
                        onkeypress="if(event.key=='Enter') sendMessage()"
                    >
                    <button id="sendBtn" onclick="sendMessage()">Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let conversationHistory = [];
        
        async function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            const messagesDiv = document.getElementById('messages');
            const sendBtn = document.getElementById('sendBtn');
            
            // Add user message
            const userMsg = document.createElement('div');
            userMsg.className = 'message user';
            userMsg.textContent = message;
            messagesDiv.appendChild(userMsg);
            
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Add to history
            conversationHistory.push({
                role: 'user',
                content: message
            });
            
            // Add thinking message
            const thinkingMsg = document.createElement('div');
            thinkingMsg.className = 'message assistant';
            thinkingMsg.textContent = 'Thinking...';
            messagesDiv.appendChild(thinkingMsg);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            try {
                const response = await fetch('/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        message: message,
                        history: conversationHistory
                    })
                });
                
                const data = await response.json();
                thinkingMsg.textContent = data.response;
                
                // Add to history
                conversationHistory.push({
                    role: 'assistant',
                    content: data.response
                });
                
            } catch (err) {
                thinkingMsg.textContent = 'Error: Unable to get response. Please try again.';
                thinkingMsg.style.color = '#d32f2f';
            }
            
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function clearChat() {
            if (confirm('Clear the conversation?')) {
                conversationHistory = [];
                const messagesDiv = document.getElementById('messages');
                messagesDiv.innerHTML = `
                    <div class="message assistant">
                        Hello! I'm your AI assistant. How can I help you today?
                    </div>
                `;
            }
        }
        
        // Focus on input
        document.getElementById('messageInput').focus();
    </script>
</body>
</html>
""".replace('{{ MODEL }}', MODEL)


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '')
        history = data.get('history', [])
        
        if not message:
            return jsonify({'response': 'Please enter a message.'})
        
        print(f"\nUser: {message}")
        
        # Build conversation context
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        # Add history (keep last 10 exchanges)
        for msg in history[-20:]:
            messages.append(msg)
        
        # Get response from Ollama
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": MODEL,
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        assistant_message = result['message']['content']
        
        print(f"Assistant: {assistant_message}\n")
        
        return jsonify({'response': assistant_message})
    
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({
            'response': "I apologize, but I'm having trouble processing your request right now."
        })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting LLM Chat on http://0.0.0.0:8080")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=8080, debug=False)

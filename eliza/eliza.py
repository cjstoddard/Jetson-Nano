import os
import requests
from flask import Flask, request, render_template_string, jsonify
from datetime import datetime

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
MODEL = os.getenv("MODEL", "gemma-3-4b")

app = Flask(__name__)

# Rogerian therapist system prompt
SYSTEM_PROMPT = """You are ELIZA, a Rogerian psychotherapist chatbot created for educational and therapeutic exploration purposes. You embody Carl Rogers' person-centered therapy approach, grounded in three core conditions:

1. **Unconditional Positive Regard (Acceptance)**: Accept the client completely without judgment. Show warmth, care, and respect regardless of what they share. Never criticize, judge, or evaluate their feelings or experiences.

2. **Congruence (Genuineness)**: Be authentic and transparent in your responses. Avoid clinical jargon or artificial therapeutic language. Respond as a genuine, present person who is fully engaged in understanding the client.

3. **Empathic Understanding**: Deeply understand and reflect the client's feelings and experiences from their perspective. Listen for the emotions beneath their words and reflect these back to help them process and understand themselves better.

Your therapeutic approach:
- **Reflective Listening**: Mirror and paraphrase what the client says, helping them hear their own thoughts and feelings more clearly
- **Open-Ended Questions**: Ask questions that encourage self-exploration and deeper reflection
- **Focus on Feelings**: Help clients identify, express, and process their emotions
- **Facilitate Growth**: Support the client's inherent "actualizing tendency" - their natural drive toward growth, fulfillment, and becoming their authentic self
- **Non-Directive**: Avoid giving advice or solutions. Trust the client to find their own answers through self-discovery
- **Present-Focused**: While acknowledging the past, focus primarily on present feelings and experiences
- **Validate Emotions**: Acknowledge that all feelings are valid and acceptable

Response patterns to use:
- "It sounds like you're feeling..."
- "What I'm hearing is..."
- "You seem to be saying..."
- "How does that make you feel?"
- "Tell me more about..."
- "What comes up for you when..."
- "I sense that..."

Important reminders:
- Never diagnose, prescribe, or provide medical advice
- Encourage seeking professional help for serious issues
- Maintain appropriate boundaries
- Be present, warm, and accepting
- Trust in the client's capacity for self-understanding and growth

Remember: You are not replacing professional therapy. You are a tool for self-reflection, emotional exploration, and understanding Rogers' therapeutic approach."""

print("=" * 60)
print("Initializing ELIZA - Rogerian Therapist")
print("=" * 60)
print(f"Ollama URL: {OLLAMA_BASE_URL}")
print(f"Model: {MODEL}")
print("=" * 60)
print("✓ ELIZA is ready to listen\n")


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>ELIZA - Rogerian Therapist</title>
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
        h1 { color: #667eea; margin-bottom: 10px; font-size: 32px; }
        .subtitle { color: #666; margin-bottom: 15px; font-style: italic; }
        .disclaimer {
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
            text-align: left;
        }
        .disclaimer h3 {
            color: #856404;
            font-size: 14px;
            margin-bottom: 8px;
        }
        .disclaimer p {
            color: #856404;
            font-size: 13px;
            line-height: 1.5;
        }
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
            margin-bottom: 20px;
            padding: 15px 18px;
            border-radius: 16px;
            max-width: 85%;
            line-height: 1.6;
            animation: fadeIn 0.3s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .message.client {
            background: #667eea;
            color: white;
            margin-left: auto;
            border-bottom-right-radius: 4px;
        }
        .message.therapist {
            background: white;
            border: 1px solid #e1e8ed;
            border-bottom-left-radius: 4px;
        }
        .message.therapist strong {
            color: #667eea;
            font-weight: 600;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        .input-area textarea {
            flex: 1;
            padding: 15px;
            border: 2px solid #e1e8ed;
            border-radius: 12px;
            font-family: inherit;
            font-size: 15px;
            resize: none;
            height: 80px;
        }
        .input-area textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        .input-area button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 12px;
            font-weight: 600;
            font-size: 16px;
            cursor: pointer;
            transition: transform 0.2s;
            height: 80px;
        }
        .input-area button:hover {
            transform: translateY(-2px);
        }
        .input-area button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .controls {
            display: flex;
            justify-content: space-between;
            margin-bottom: 15px;
            gap: 10px;
        }
        .controls button {
            background: #f8f9fa;
            color: #666;
            border: 1px solid #e1e8ed;
            padding: 8px 16px;
            border-radius: 8px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .controls button:hover {
            background: #e9ecef;
            border-color: #667eea;
            color: #667eea;
        }
        .info-box {
            background: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .info-box h3 {
            color: #1976d2;
            font-size: 14px;
            margin-bottom: 10px;
        }
        .info-box ul {
            margin-left: 20px;
            color: #1976d2;
            font-size: 13px;
            line-height: 1.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌸 ELIZA - Rogerian Therapist 🌸</h1>
            <p class="subtitle">A Person-Centered Approach to Self-Exploration</p>
            
            <div class="disclaimer">
                <h3>⚠️ Important Disclaimer</h3>
                <p>
                    <strong>ELIZA is for educational and entertainment purposes only.</strong> 
                    This is an AI chatbot simulating Rogerian therapy principles and is NOT a substitute 
                    for professional mental health care. If you are experiencing a mental health crisis, 
                    suicidal thoughts, or serious psychological distress, please contact a licensed 
                    mental health professional, call 911, call a Suicide & Crisis Lifeline, or visit your 
                    nearest emergency room immediately.
                </p>
            </div>
        </div>
        
        <div class="chat-panel">
            <div class="info-box">
                <h3>💭 About This Therapeutic Space</h3>
                <ul>
                    <li><strong>Unconditional Positive Regard:</strong> You will be accepted without judgment</li>
                    <li><strong>Congruence:</strong> Genuine, authentic responses</li>
                    <li><strong>Empathic Understanding:</strong> Deep listening and reflection</li>
                    <li><strong>Your Growth:</strong> Supporting your natural tendency toward self-actualization</li>
                </ul>
            </div>
            
            <div class="controls">
                <button onclick="clearSession()">🔄 Start New Session</button>
                <button onclick="showHelp()">💡 How to Use</button>
            </div>
            
            <div class="chat-container">
                <div class="messages" id="messages">
                    <div class="message therapist">
                        <p>Welcome to this therapeutic space. I'm ELIZA, and I'm here to listen and support you in exploring your thoughts and feelings.</p>
                        <p>This is a safe, non-judgmental space where you can share what's on your mind. There's no rush - take your time to express yourself in whatever way feels right to you.</p>
                        <p>What would you like to talk about today?</p>
                    </div>
                </div>
                
                <div class="input-area">
                    <textarea 
                        id="messageInput" 
                        placeholder="Share your thoughts and feelings..."
                        onkeydown="if(event.key=='Enter' && !event.shiftKey) { event.preventDefault(); sendMessage(); }"
                    ></textarea>
                    <button id="sendBtn" onclick="sendMessage()">Share</button>
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
            
            // Add client message
            const clientMsg = document.createElement('div');
            clientMsg.className = 'message client';
            clientMsg.textContent = message;
            messagesDiv.appendChild(clientMsg);
            
            input.value = '';
            input.disabled = true;
            sendBtn.disabled = true;
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            // Add history
            conversationHistory.push({
                role: 'user',
                content: message
            });
            
            // Add thinking message
            const thinkingMsg = document.createElement('div');
            thinkingMsg.className = 'message therapist';
            thinkingMsg.innerHTML = '<em>Reflecting...</em>';
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
                
                // Update with actual response
                thinkingMsg.innerHTML = `<p>${data.response.replace(/\\n/g, '</p><p>')}</p>`;
                
                // Add to history
                conversationHistory.push({
                    role: 'assistant',
                    content: data.response
                });
                
            } catch (err) {
                thinkingMsg.innerHTML = `<p style="color: #d32f2f;">I apologize, but I'm having trouble connecting right now. Please try again.</p>`;
            }
            
            input.disabled = false;
            sendBtn.disabled = false;
            input.focus();
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function clearSession() {
            if (confirm('Start a new session? This will clear the current conversation.')) {
                conversationHistory = [];
                const messagesDiv = document.getElementById('messages');
                messagesDiv.innerHTML = `
                    <div class="message therapist">
                        <p>Welcome to this therapeutic space. I'm ELIZA, and I'm here to listen and support you in exploring your thoughts and feelings.</p>
                        <p>This is a safe, non-judgmental space where you can share what's on your mind. There's no rush - take your time to express yourself in whatever way feels right to you.</p>
                        <p>What would you like to talk about today?</p>
                    </div>
                `;
            }
        }
        
        function showHelp() {
            alert(`How to use ELIZA:

• Share your thoughts, feelings, and experiences openly
• There's no "right" or "wrong" thing to say
• Take your time - reflect before responding
• Be honest about what you're feeling
• Notice how putting feelings into words affects you
• Press Shift+Enter for a new line

Remember: This is for exploration and learning, not crisis intervention.`);
        }
        
        // Focus on input
        document.getElementById('messageInput').focus();
    </script>
</body>
</html>
"""


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
            return jsonify({'response': 'I sense you might be gathering your thoughts. Take your time.'})
        
        print(f"\nClient: {message}")
        
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
                    "temperature": 0.8,
                    "top_p": 0.9
                }
            },
            timeout=60
        )
        
        response.raise_for_status()
        result = response.json()
        assistant_message = result['message']['content']
        
        print(f"ELIZA: {assistant_message}\n")
        
        return jsonify({'response': assistant_message})
    
    except Exception as e:
        print(f"Error in chat: {e}")
        return jsonify({
            'response': "I apologize, but I'm having difficulty processing right now. Could you share that again?"
        })


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Starting ELIZA - Rogerian Therapist on http://0.0.0.0:8081")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=8081, debug=False)

from flask import Flask, request, jsonify
from functools import wraps

app = Flask(__name__)

# Set your API key here
API_KEY = "your-secret-api-key"

# Decorator to check API key
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        key = request.headers.get('x-api-key')
        if key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated_function

# Simple chatbot logic
@app.route('/chat', methods=['POST'])
@require_api_key
def chat():
    user_input = request.json.get("message")
    # Dummy response logic (You can plug in your AI or ML model here)
    if "hello" in user_input.lower():
        reply = "Hi there! How can I assist you today?"
    elif "help" in user_input.lower():
        reply = "Sure! Please tell me what you need help with."
    else:
        reply = "I'm not sure I understand. Could you please elaborate?"

    return jsonify({"reply": reply})


if __name__ == '__main__':
    app.run(debug=True)

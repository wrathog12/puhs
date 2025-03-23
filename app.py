from flask import Flask, request, jsonify
from chatbot import get_answer  # Import the get_answer function from your chatbot.py
from dotenv import load_dotenv
import os

# ✅ Load environment variables
load_dotenv()

# ✅ Initialize Flask app
app = Flask(__name__)

# ✅ Define route for chatting
@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Get the message sent from the frontend (user input)
        data = request.json
        user_query = data.get("message", "")

        # Ensure the user input is not empty
        if not user_query:
            return jsonify({"error": "Empty message. Please send a valid query."}), 400

        # Get the answer from your chatbot using the get_answer function
        response = get_answer(user_query)

        # Return the response in JSON format
        return jsonify({
            "source": "FAISS + Gemini",  # Optionally, you can define where the response came from
            "answer": response
        })

    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500

# ✅ Run Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

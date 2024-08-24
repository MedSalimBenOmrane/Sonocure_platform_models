from flask import Flask, request, jsonify
from flask_cors import CORS 
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Configure the API key for Google Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize the chatbot model
model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

# Create a Flask app
app = Flask(__name__)
CORS(app)
# Define a function to get a response from the Gemini chatbot
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    full_response = ''.join([chunk.text for chunk in response])
    return full_response

# Define a route for the chatbot API
@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the question from the request
    data = request.json
    question = data.get('question')

    # Get the chatbot's response
    response = get_gemini_response(question)

    # Return the response as JSON
    return jsonify({'response': response})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
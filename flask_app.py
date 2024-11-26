from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib

app = Flask(__name__, static_folder="static", template_folder="static")
CORS(app)

# Load the LSTM model and tokenizer
model = load_model('lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

@app.route('/')
def home():
    """Serve the HTML frontend."""
    return app.send_static_file('index.html')

@app.route('/classify', methods=['POST'])
def classify_comment():
    try:
        # Get the input comment
        data = request.json
        comment = data.get('comment', '')

        # Preprocess the input text
        tokenized_comment = tokenizer.texts_to_sequences([comment])
        tokenized_comment = pad_sequences(tokenized_comment, maxlen=150)

        # Make prediction
        prediction = model.predict(tokenized_comment)
        result = "Cyberbullying" if prediction[0][0] > 0.5 else "Not Cyberbullying"

        # Return the result
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)

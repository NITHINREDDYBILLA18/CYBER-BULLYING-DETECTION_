import gradio as gr
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the LSTM model and tokenizer
model = load_model('lstm_model.h5')
tokenizer = joblib.load('tokenizer.pkl')

# Define the prediction function
def classify_comment(comment):
    # Tokenize and preprocess the input comment
    tokenized_comment = tokenizer.texts_to_sequences([comment])
    tokenized_comment = pad_sequences(tokenized_comment, maxlen=150)

    # Make prediction
    prediction = model.predict(tokenized_comment)
    result = "Cyberbullying" if prediction[0][0] > 0.5 else "Not Cyberbullying"
    return result

# Create the Gradio interface
interface = gr.Interface(
    fn=classify_comment,  # The function to call when input is provided
    inputs=gr.Textbox(lines=2, placeholder="Enter a comment..."),
    outputs="text",
    title="Cyberbullying Detection",
    description="Enter a comment to classify it as Cyberbullying or Not Cyberbullying"
)

# Launch the Gradio app
if __name__ == "__main__":
    interface.launch()

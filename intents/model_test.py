import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load tokenizer
with open("model_files/tokenizer.pkl", "rb") as handle:
    tokenizer = pickle.load(handle)

# Load label encoder
with open("model_files/label_encoder.pkl", "rb") as handle:
    label_encoder = pickle.load(handle)

# Load class names
with open("model_files/classes.pkl", "rb") as handle:
    classes = pickle.load(handle)

# Load trained model
model = load_model("model_files/intents.h5")  # Ensure correct path

def predict_intent(text):
    """Preprocesses input text, runs model prediction, and returns the predicted intent."""
    # Tokenize and pad input text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20)  # Adjust maxlen based on training

    # Predict using the trained model
    pred = model.predict(padded)
    intent_idx = np.argmax(pred)  # Get highest confidence index

    # Decode intent
    intent_label = label_encoder.inverse_transform([intent_idx])[0]
    return intent_label

def main():
    """Runs an interactive chatbot loop."""
    print("Chatbot is running. Type 'quit' or 'exit' to stop.")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit"]:
            print("Exiting chatbot.")
            break
        
        intent = predict_intent(user_input)
        print(f"Predicted intent: {intent}")

# Run the main function only if this script is executed directly
if __name__ == "__main__":
    main()

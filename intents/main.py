import logging
import pickle
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

app = FastAPI(
    title="Intent Classifier API",
    description="An API that classifies user queries into intents using a pre-trained model.",
    version="1.0.0",
)

# Pydantic models for request and response
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    intent: str

# Load model files on startup
try:
    with open("model_files/rephrased_tokenizer.pkl", "rb") as handle:
        tokenizer = pickle.load(handle)
    logging.info("Tokenizer loaded successfully.")

    with open("model_files/rephrased_label_encoder.pkl", "rb") as handle:
        label_encoder = pickle.load(handle)
    logging.info("Label encoder loaded successfully.")

    with open("model_files/rephrased_classes.pkl", "rb") as handle:
        classes = pickle.load(handle)
    logging.info(f"Classes loaded successfully: {classes}")

    model = load_model("model_files/rephrased_intents.h5")
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model files: %s", e)
    raise e

def predict_intent(text: str) -> str:
    """Preprocess the input text, run prediction, and decode the predicted intent."""
    # Tokenize and pad input text
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=20)  # Adjust maxlen as per your training settings
    logging.info(f"Tokenized input: {seq}")
    logging.info(f"Padded input: {padded}")

    # Predict using the model
    pred = model.predict(padded)
    logging.info(f"Raw model output: {pred}")

    # Get index of highest probability and decode the intent
    intent_idx = np.argmax(pred)
    intent_label = label_encoder.inverse_transform([intent_idx])[0]
    logging.info(f"Predicted intent: {intent_label}")

    return intent_label

@app.post("/classify", response_model=QueryResponse)
async def classify_query(request: QueryRequest):
    try:
        intent = predict_intent(request.query)
        return QueryResponse(intent=intent)
    except Exception as e:
        logging.error("Error during classification: %s", e)
        raise HTTPException(status_code=500, detail=f"Error processing query: {e}")

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "API is running."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from classifier import CustomModel

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
    tokenizer = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
    logging.info("Tokenizer loaded successfully.")
    config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
    logging.info("Config loaded successfully.")
    model = CustomModel(
        target_sizes=config.target_sizes,
        task_type_map=config.task_type_map,
        weights_map=config.weights_map,
        divisor_map=config.divisor_map,
    )
    model.load_state_dict(torch.load("model_files/trained_model.pth", map_location=torch.device("cpu")))
    model.eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error("Error loading model files: %s", e)
    raise e

def predict_intent(text: str) -> str:
    """Preprocess the input text, run prediction, and decide route label based on complexity"""
    # Tokenize and pad input text
    encoded_text = tokenizer(text, 
                            return_tensors="pt",
                            add_special_tokens=True,
                            max_length=512,
                            padding="max_length",
                            truncation=True,
                            )
    logging.info(f"Tokenized input: {encoded_text}")
    
    # Predict using the model
    pred = model(encoded_text)
    logging.info(f"Raw model output: {pred}")
    if pred['prompt_complexity_score'][0] > 0.25:
        intent_label = "complex"
    else:
        intent_label = "simple"
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

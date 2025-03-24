import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from classifier import CustomModel

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")

# Load model config
config = AutoConfig.from_pretrained("nvidia/prompt-task-and-complexity-classifier")

# Load trained model
model = CustomModel(
    target_sizes=config.target_sizes,
    task_type_map=config.task_type_map,
    weights_map=config.weights_map,
    divisor_map=config.divisor_map,
)

# Load model weights
model.load_state_dict(torch.load("model_files/trained_model.pth", map_location=torch.device("cpu")))
model.eval()

def predict_intent(text):
    """Preprocesses input text, runs model prediction, and returns the predicted intent."""
    # Tokenize and pad input text
    encoded_text = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=True,
        max_length=512,
        padding="max_length",
        truncation=True,
    )
    
    # Predict using the trained model
    result = model(encoded_text)
    if result['prompt_complexity_score'][0] > 0.25:
        intent_label = "complex"
    else:
        intent_label = "simple"
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

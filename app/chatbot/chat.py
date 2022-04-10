import os
import json
import random
import torch 
from .model import ChatbotNet 
from .utils import bag_of_words, tokenize_sentence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SERIALIZED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "serialized_data.pth")
INTENTS_FILE_PATH = os.path.join(os.path.dirname(__file__),"intents.json")

intents = json.load(open(INTENTS_FILE_PATH,"r+"))["intents"]

serialized_data = torch.load(SERIALIZED_MODEL_PATH)
input_size = serialized_data["input_size"]
hidden_size = serialized_data["hidden_size"]
n_classes = serialized_data["n_classes"]
all_words = serialized_data["all_words"]
tags = serialized_data["tags"]
model_state = serialized_data["model_state"]

model = ChatbotNet(
    input_size=input_size,
    hidden_size=hidden_size, 
    num_classes=n_classes 
)
model.load_state_dict(model_state)

# set to evaluation mode 
model.eval()

def get_response(message:str, confidence:int=0.7)->str:
    tokenized_message = tokenize_sentence(message)
    X = bag_of_words(tokenized_message, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X)
    output = model(X)
    # obtain the prediction probabilities
    probas = torch.softmax(output, dim=1)
    pred_idx = probas.argmax(dim=1).item()
    pred_proba = probas[0][pred_idx]
    pred_tag = tags[pred_idx]
    if pred_proba < confidence: 
        return "I do not understand."
    intent = list(filter(lambda x: x["tag"]==pred_tag, intents))[0]
    return random.choice(intent["responses"])


if __name__ == "__main__":
    bot_name="Bot"
    print("Let's chat! Type 'quit' to exit.")
    while True: 
        sentence = input("You: ")
        if sentence.lower() == "quit":
            break 

        sentence = tokenize_sentence(sentence)
        X = bag_of_words(sentence, all_words)
        X = X.reshape(1, X.shape[0])
        X = torch.from_numpy(X)
        output = model(X)
        # print(output)
        _, predicted = torch.max(output, dim=1)
        predicted_tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs.max().item()

        if prob > 0.7:
            intent = list(filter(lambda x: x["tag"]==predicted_tag, intents))[0]
            print(f"{bot_name}: {random.choice(intent['responses'])}")
        else: 
            print(f"{bot_name}: I do not understand.")
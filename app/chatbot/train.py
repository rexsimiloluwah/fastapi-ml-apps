import json 
import torch 
import numpy as np
from torch import nn 
from torch.utils.data import Dataset, DataLoader
from utils import tokenize_sentence, stem, bag_of_words
from model import ChatbotNet 

intents = json.load(open("intents.json","r+"))["intents"]
all_words = []
tags = []
train_data = []

for intent in intents:
    tag = intent["tag"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize_sentence(pattern)
        all_words.extend(tokenize_sentence(pattern))
        train_data.append((w,tag))

punctuation_chars = "/\?.!-|~`':><,*&^$#@_+="
all_words = [stem(w) for w in all_words if w not in punctuation_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []
for (patterns_sentence, tag) in train_data: 
    # using bag of words for the word-level representation 
    bag = bag_of_words(patterns_sentence, all_words)
    X_train.append(bag)
    label = tags.index(tag)
    y_train.append(label)

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.float32)

class ChatbotDataset(Dataset):
    def __init__(self,X_train:np.ndarray,y_train:np.ndarray):
        super(ChatbotDataset,self).__init__()
        self.n_samples = len(X_train)
        self.x_data = X_train 
        self.y_data = y_train 
    
    def __getitem__(self,index:int):
        return (self.x_data[index], self.y_data[index]) 
    
    def __len__(self):
        return self.n_samples 

dataset = ChatbotDataset(X_train, y_train)

# Define Hyperparameters
BATCH_SIZE = 8
INIT_LR = 1e-3 
N_EPOCHS = 1000
hidden_size = 10 
n_classes = len(tags)
input_size = X_train.shape[1]

train_data_loader = DataLoader(
    dataset=dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

model = ChatbotNet(
    input_size=input_size, 
    hidden_size=hidden_size,
    num_classes=n_classes,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Loss and Optimizer 
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=INIT_LR)

# Training loop 
for epoch in range(N_EPOCHS):
    total_correct_predictions, total_loss = 0,0
    for batch_idx, (X,y) in enumerate(train_data_loader):
        X,y = X.to(device), y.to(device).type(torch.LongTensor)
        size = len(X)
        num_batches = len(train_data_loader)

        preds = model(X)
        loss = loss_fn(preds, y)
        # print(preds,y)
        # Backward propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    if (epoch+1)%10 == 0:
        print(f"Epoch {epoch}/{N_EPOCHS}: loss: {loss.item():>.5f}")

print("====="*5)
print(f"Final loss: {total_loss/num_batches:>.5f}")

# serializing the model 
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size, 
    "n_classes": n_classes, 
    "all_words": all_words,
    "tags": tags,
}

OUTPUT_FILE = "serialized_data.pth"
torch.save(data, OUTPUT_FILE)

print("Completely trained and serialized model.")

if __name__ == "__main__": 
    # print("All words: ", all_words)
    # print("Tags: ", tags)
    # print("Length of all words: ",len(all_words))
    # print("Length of tags: ", len(tags))
    # print("Training data: ", train_data)
    # print(model)
    pass
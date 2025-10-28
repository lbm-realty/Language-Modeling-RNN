import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

# Reading the file
try: 
    with open("random_text.txt", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    print("File not found")

# Creating a dictionary of unique words and creating an array to store all the words
word = ""
words = []
dictionary = {}
num = 1
for letter in text:
    if letter == " " or letter == "\n" or letter in ".?/,}{][)(=+-_*&^%$#@!;:":
        if word == "":
            continue
        if word.lower() not in dictionary:
            dictionary[word.lower()] = num 
            num += 1
        words.append(word.lower())
        word = ""
    else:
        word += letter    

# Encoding words to numerical values
encoded_text = []
for word in words:
    encoded_text.append(dictionary[word])
    
# Converting the text to sequences of 30 each -> helps with training
sequence_size = 30
inputs, targets = [], []
for i in range(len(encoded_text) - sequence_size):
    inputs.append(encoded_text[i:i+sequence_size])
    targets.append(encoded_text[i+1:i+sequence_size+1])

encoded_text = torch.tensor(encoded_text)
inputs = torch.tensor(inputs)
targets = torch.tensor(targets)

# Creating a dataset class for better data processing
class CustomTextDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

training_data = CustomTextDataset(inputs, targets)
# Loading the data, keeping shuffle False for now
train_loader = DataLoader(training_data, batch_size=2, shuffle=False)

# Model building
class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        out = self.fc(out)
        return out

vocab_size = len(dictionary) + 1
embed_size = 64
hidden_size = 64
MyRNN = SimpleRNN(vocab_size, embed_size, hidden_size)
optimizier = optim.Adam(MyRNN.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
epochs = 1000
tar_p, out_p = [], []
for epoch in range(epochs):
    for inp, tar in train_loader:
        optimizier.zero_grad()
        output = MyRNN(inp)
        output = output.view(-1, vocab_size)
        tar = tar.view(-1)
        loss = criterion(output, tar)
        loss.backward()
        optimizier.step()
    tar_p, out_p = tar, output
    if epoch % 100 == 0: print(f"At epoch {epoch}, loss: {loss}")

torch.save(MyRNN.state_dict(), "my_rnn_model.pth")


import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

try: 
    with open("random_text.txt", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    print("File not found")

word = ""
dictionary = {}
num = 1
for letter in text:
    if letter == " " or letter == "\n" or letter in ".?/,}{][)(=+-_*&^%$#@!;:":
        if word.lower() not in dictionary:
            dictionary[word.lower()] = num 
            num += 1
        word = ""
    else:
        word += letter

input, target = [], []
for i, j in dictionary.items():
    if j > 0 and j < len(dictionary): 
        input.append(i)
    if j > 1 and j < len(dictionary) + 1:
        target.append(i)         

sequence_size = 30
text_embedded = []
word = ""
for letter in text:
    if letter == " " or letter == "\n" or letter in ".?/,}{][)(=+-_*&^%$#@!;:":
        text_embedded.append(dictionary[word.lower()])
        word = ""
    else:
        word += letter

sequence_size = 30
text_embedded_tensor = []
i = 0
while i < len(text_embedded):
    end_index = int(sequence_size * ((i / sequence_size) + 1))
    text_embedded_tensor.append(text_embedded[i:end_index])
    i = end_index

for arr in text_embedded_tensor:
    if len(arr) < 30:
        while len(arr) < 30:
            arr.append(-1)

text_embedded_tensor = torch.tensor(text_embedded_tensor)

class CustomTextDataset(Dataset):
    def __init__(self, text_data):
        self.text_data = text_data
    
    def __len__(self):
        return len(self.text_data)
    
    def __getitem__(self, idx):
        return self.text_data[idx]
    
training_data = CustomTextDataset(text_embedded_tensor)
test_data = CustomTextDataset(text_embedded_tensor)
train_loader = DataLoader(training_data, batch_size=2, shuffle=False)
test_loader = DataLoader(test_data, batch_size=2, shuffle=False)

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

MyRNN = SimpleRNN(len(dictionary), len(text_embedded_tensor), len(text_embedded_tensor))

epochs = 1
for epoch in range(epochs):
    data = train_loader
    print(data)
    prediction = MyRNN(text_embedded_tensor)
    optimizier = optim.Adam(prediction.parameters(), lr=0.001)
    optimizier.zero_grad()
    e_loss = nn.CrossEntropyLoss()
    loss = e_loss(prediction, test_data)
    loss.backward()
    optimizier.step()

    print(f"At epoch {epoch}: train_error: {loss}")


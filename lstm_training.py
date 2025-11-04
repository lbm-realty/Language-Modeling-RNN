import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim


# Combining and reading all the text files
file_paths = ["alice_wonderland.txt", "kids_stories.txt", "random_linkedin.txt", "shakespeare.txt"]
all_text = ""
try: 
    for path in file_paths:   
        with open(path, encoding="utf-8") as file:
            all_text += file.read() + " "
except FileNotFoundError:
    print("File not found")

# Creating a dictionary of unique words and creating an array to store all the words
word = ""
words = []
model_vocab = {}
num = 1
for letter in all_text:
    if letter == " " or letter == "\n" or letter in ".?/,}{][)(=+-_*&^%$#@!;:":
        if word == "":
            continue
        if word.lower() not in model_vocab:
            model_vocab[word.lower()] = num 
            num += 1
        words.append(word.lower())
        word = ""
    else:
        word += letter    

# Encoding words to numerical values
encoded_text = []
for word in words:
    encoded_text.append(model_vocab[word])
    
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
# Loading the data
train_loader = DataLoader(training_data, batch_size=2, shuffle=True)

# Model building
class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, hidden=None):
        x = self.embedding(x)

        if hidden is None:
            out, hidden = self.lstm(x)
        else:
            out, hidden = self.lstm(x, hidden)
        out = self.fc(out)
        return out, hidden

vocab_size = len(model_vocab) + 1
embed_size = 64
hidden_size = 64
MyLSTM = SimpleLSTM(vocab_size, embed_size, hidden_size)
optimizier = optim.Adam(MyLSTM.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training
epochs = 100
tar_p, out_p = [], []

# Increasing epoch by 1 to see the error for the last value
for epoch in range(epochs + 1):
    for inp, tar in train_loader:
        optimizier.zero_grad()
        output, hidden = MyLSTM(inp)
        output = output.view(-1, vocab_size)
        tar = tar.view(-1)
        loss = criterion(output, tar)
        loss.backward()
        optimizier.step()
    tar_p, out_p = tar, output
    if epoch % 10 == 0: print(f"At epoch {epoch}, loss: {loss}")

torch.save(MyLSTM.state_dict(), "lstm_50e.pth")

# Saving the training losses from terminal

"""
At epoch 0, loss: 2.6372008323669434
At epoch 10, loss: 0.03272818773984909
At epoch 20, loss: 0.029907794669270515
At epoch 30, loss: 0.11129624396562576
At epoch 40, loss: 0.05571934953331947
At epoch 50, loss: 0.0013217307860031724
At epoch 60, loss: 0.09435328096151352
At epoch 70, loss: 0.0003759579558391124
At epoch 80, loss: 0.00044325983617454767
At epoch 90, loss: 0.00043487997027114034
At epoch 100, loss: 0.04973514378070831
"""
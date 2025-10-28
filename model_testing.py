import torch as torch
import torch.nn as nn

# Reading the file
try: 
    with open("random_text.txt", encoding="utf-8") as file:
        text = file.read()
except FileNotFoundError:
    print("File not found")

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

text_length = len(encoded_text)
input = torch.tensor([encoded_text[0:text_length-1]])
target = torch.tensor([encoded_text[1:text_length]])

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
model = SimpleRNN(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load("my_rnn_model.pth"))
model.eval()
criterion = nn.CrossEntropyLoss()
output = model(input)
output = output.view(-1, vocab_size)
target = target.view(-1)
loss = criterion(output, target)

print(loss)

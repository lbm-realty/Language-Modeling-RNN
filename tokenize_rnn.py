import torch
import torch.nn as nn

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

text_embedded = torch.tensor(text_embedded)
print(text_embedded)
sequence_size = 30
text_embedded_tensor = torch.Tensor([])
i = 0
counter = 1
while i < len(text_embedded):
    text_embedded_tensor = torch.cat(text_embedded[i: counter * sequence_size])
    i = sequence_size + 1
    counter += 1

print(text_embedded_tensor)

# MyRNN = SimpleRNN(len(dictionary), len(input), len(input))
# print(MyRNN.forward(dictionary))
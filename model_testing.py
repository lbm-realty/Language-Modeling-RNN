import torch as torch
import torch.nn as nn

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
    
# Testing
vocab_size = len(model_vocab) + 1
embed_size = 64
hidden_size = 64
model = SimpleRNN(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load("my_rnn2_model.pth"))
model.eval()
criterion = nn.CrossEntropyLoss()
output = model(input)
output = output.view(-1, vocab_size)
target = target.view(-1)
loss = criterion(output, target)

print(loss)

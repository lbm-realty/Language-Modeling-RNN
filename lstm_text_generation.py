import torch as torch
import torch.nn as nn

"""
Models
    -> lstm_10e, lstm_20e, lstm_50e, lstm_100e
"""

# Combining and reading all the text files
file_paths = ["alice_wonderland.txt", "kids_stories.txt", "random_linkedin.txt", "shakespeare.txt"]
all_text = ""
try: 
    for path in file_paths:   
        with open(path, encoding="utf-8") as file:
            all_text += file.read() + " "
except FileNotFoundError:
    print("File not found")

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

unique_words_nums = [number for number in model_vocab.values()] 

# Encoding the all the text from words to numbers
encoded_text = []
for word in words:
    encoded_text.append(model_vocab[word])

# Converting encoded_text list to tensor
input_text_encoded = torch.tensor([encoded_text])

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
    
# Testing
vocab_size = len(model_vocab) + 1
embed_size = 64
hidden_size = 64
model = SimpleLSTM(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load("lstm_50e.pth"))
model.eval()
criterion = nn.CrossEntropyLoss()
reduced_input = torch.tensor(input_text_encoded[0, -30:]).unsqueeze(0) 
output, hidden = model(reduced_input)

num_words = input("How many words do you want to generate: ")
idx_to_word = {v: k for k, v in model_vocab.items()}

for _ in range(int(num_words)):
    output, hidden = model(reduced_input)
    next_word_idx = torch.argmax(output[0, -1]).item()
    print(idx_to_word[next_word_idx], end=" ")
    reduced_input = reduced_input.squeeze(0).tolist() + [next_word_idx]
    reduced_input = torch.tensor(reduced_input[-30:]).unsqueeze(0)
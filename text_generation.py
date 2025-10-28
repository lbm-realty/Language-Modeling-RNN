import torch
import torch.nn as nn

# Reading the file and enocding the text
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

unique_words_nums = [number for number in dictionary.values()] 

encoded_text = []
for word in words:
    encoded_text.append(dictionary[word])

text_length = len(encoded_text)
input_text_encoded = torch.tensor([encoded_text[0:text_length-1]])
target = torch.tensor([encoded_text[-30]])

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
vocab_size = len(dictionary) + 1
embed_size = 64
hidden_size = 64
model = SimpleRNN(vocab_size, embed_size, hidden_size)
model.load_state_dict(torch.load("my_rnn_model.pth"))
model.eval()
criterion = nn.CrossEntropyLoss()
reduced_input = torch.tensor(input_text_encoded[0, -30:])
output = model(reduced_input)

num_words = input("How many words do you want to generate: ")

for _ in range(int(num_words) + 1):
    output = model(reduced_input)
    highest_prob_idx = torch.argmax(output[-1]).item()
    unique_words = list(dictionary.keys())
    print(unique_words[highest_prob_idx], end=" ")
    gen_word = unique_words_nums[highest_prob_idx]
    reduced_input = reduced_input.tolist()
    reduced_input.append(gen_word)
    reduced_input = torch.tensor(reduced_input[-30:])

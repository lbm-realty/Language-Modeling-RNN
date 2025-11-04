import torch
import torch.nn as nn
import pickle

with open("rnn2_vocab.pkl", "rb") as f:
    model_vocab = pickle.load(f)

idx_to_word = {v: k for k, v in model_vocab.items()}

def generate_text(input_seq, sequence_size, num_of_words, model):
    model.eval()
    hidden = None
    generated_text = []
    input_seq = input_seq[-sequence_size:]
    input_tensor = torch.tensor([input_seq])
    for _ in range(num_of_words):
        if isinstance(model, SimpleLSTM):
            output, hidden = model(input_tensor, hidden)
        else:
            output = model(input_tensor)

        next_idx = torch.argmax(output[0, -1]).item()
        next_word = idx_to_word[next_idx]
        input_seq = input_seq[1:] + [next_idx]
        input_tensor = torch.tensor([input_seq])
        generated_text.append(next_word)
    
    return generated_text

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


vocab_size = len(model_vocab) + 1
embed_size = 64
hidden_size = 64
sequence_size = 30

# Load models
rnn_model = SimpleRNN(vocab_size, embed_size, hidden_size)
rnn_model.load_state_dict(torch.load("rnn_20e.pth"))
lstm_model = SimpleLSTM(vocab_size, embed_size, hidden_size)
lstm_model.load_state_dict(torch.load("lstm_20e.pth"))

prompt = "The cat was on the"
encoded_prompt = [model_vocab.get(word) for word in prompt.lower().split()]

words_to_gen = int(input("How many words do you want to generate? "))

rnn_generation = generate_text(encoded_prompt, sequence_size, words_to_gen, rnn_model)
lstm_generation = generate_text(encoded_prompt, sequence_size, words_to_gen, lstm_model)

print("RNN output for 20 epochs:")
print(" ".join(prompt.split() + rnn_generation))
print("\nLSTM output for 20 epochs:")
print(" ".join(prompt.split() + lstm_generation))

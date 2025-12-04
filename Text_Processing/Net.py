import torch
import torch.nn as nn

class TextGeneratorRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(TextGeneratorRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,     
            hidden_size=hidden_dim,       
            num_layers=num_layers,         
            dropout=dropout if num_layers > 1 else 0, 
            batch_first=True              
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        output = self.fc(lstm_out)
        return output, hidden
    
    def predict_next_char(self, x, hidden=None, temperature=0.8):
        output, hidden = self.forward(x, hidden)
        logits = output[0, -1, :]
        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        next_char_id = torch.multinomial(probabilities, num_samples=1).item()
        return next_char_id, hidden


class TextGeneratorGRU(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.3):
        super(TextGeneratorGRU, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        embedded = self.dropout(embedded)
        gru_out, hidden = self.gru(embedded, hidden)
        gru_out = self.dropout(gru_out)
        output = self.fc(gru_out)
        return output, hidden
    
    def predict_next_char(self, x, hidden=None, temperature=0.8):
        output, hidden = self.forward(x, hidden)
        logits = output[0, -1, :]
        scaled_logits = logits / temperature
        probabilities = torch.softmax(scaled_logits, dim=-1)
        next_char_id = torch.multinomial(probabilities, num_samples=1).item()
        return next_char_id, hidden


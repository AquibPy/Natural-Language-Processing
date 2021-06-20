import torch.nn as nn

class WordLSTM(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        # input_dim <--- vocabulary size
        # output_dim <--- len ([positive, negative]) == 2 
        # emb_dim <--- embedding dimension of embedding matrix
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.lstm = nn.LSTM(emb_dim,hidden_dim,n_layers,dropout = dropout)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc2 = nn.Linear(hidden_dim//2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        embedded = self.dropout(self.embedding(x))
        out, (hidden,cell) = self.lstm(embedded)
        # output shape -> [batch, hidden_dim]
        # hiddden shape -> [n_layers, batch, hidden_dim]
        # cell shape -> [n_layers, batch, hidden_dim]
        out = self.fc1(out[-1])
        out = self.fc2(self.relu(out))
        return out
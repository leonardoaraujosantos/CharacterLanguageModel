import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def initHidden(batch_size, bidirectional, hidden_size, num_layers, device):
    if bidirectional:
            num_directions=2
    else:
        num_directions=1
    return torch.zeros(num_layers * num_directions, batch_size, hidden_size, device=device)


# Simple Character Language Model
class CharLangModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(CharLangModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers         
        self.output_size = output_size

        self.embedding = nn.Embedding(input_size, hidden_size)        
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden, input_len):                
        # pack_padded_sequence so that padded items in the sequence won't be shown to the LSTM      
        # The default order is (seq_len, batch, input_size)        
        batch_size, seq_len = input.size()        
        embedded = self.embedding(input)
        embedded = F.relu(embedded)        
        # Pack sequence B x T x * where T is the length of the longest sequence and B is the batch size
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_len, enforce_sorted=False, batch_first=True)
        # The default order is (seq_len, batch, input_size)
        output, hidden = self.gru(packed, hidden)
        # unpack (back to padded)        
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True, total_length=seq_len)         
        # (batch_size, seq_len, nb_lstm_units) -> (batch_size * seq_len, nb_lstm_units)
        output = output.contiguous()
        output = output.view(-1, output.shape[2])        
        
        #output = self.softmax(self.out(output)).squeeze(0)
        # Use F.Softmax for CrossEntropy and LogSoftmax for NLLLoss        
        linear_out = self.out(output)        
        #print('linear_out shape:', linear_out.shape)
        # Use F.Softmax for CrossEntropy and LogSoftmax for NLLLoss
        output = F.softmax(linear_out, dim=1)        
        #output = self.softmax(linear_out).squeeze(0)
        # (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, output_size)
        output = output.view(batch_size, seq_len, self.output_size)        
        
        return output, hidden
    
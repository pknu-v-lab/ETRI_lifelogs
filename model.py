import torch.nn as nn
import torch

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Sequential(
                   nn.Linear(hidden_size, output_size),
                   nn.Sigmoid()         
        )
        self.dropout = nn.Dropout(0.3)  
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        out = self.fc(output[:, -1, :])  
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
                   nn.Linear(hidden_size * 2, output_size),
                   nn.Sigmoid(),
        )
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  
        
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device) 
        c0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        packed_output, _ = self.lstm(packed_input, (h0, c0))
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        
        out = self.fc(output[:, -1, :])  
        return out
    
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid(),
        )
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        packed_output, _ = self.gru(packed_input, h0)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output[:, -1, :])
        return out
    
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=0.5, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, output_size), 
            nn.Sigmoid(),
        )
    
    def forward(self, x, lengths):
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)
        
        packed_input = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        
        h0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(x.device)
        packed_output, _ = self.gru(packed_input, h0)
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output[:, -1, :])  
        return out
    
class Ensemble_Model(nn.Module):
    
    def __init__(self, model_class, model_num, input_size, hidden_size, num_layers, output_size, combined_weight_path):
        super(Ensemble_Model, self).__init__()
        self.models = nn.ModuleList()
        
        if torch.cuda.is_available():
            combined_state_dict = torch.load(combined_weight_path)
        else:
            combined_state_dict = torch.load(combined_weight_path, map_location='cpu')
        
        if model_num == 1 :
            self.models.append(model_class(input_size, hidden_size, num_layers, output_size))
            
            if torch.cuda.is_available():
                self.models[0].load_state_dict(torch.load(combined_weight_path))
            else:
                self.models[0].load_state_dict(torch.load(combined_weight_path, map_location='cpu'))
            
        else:
            for i in range(model_num):
                if i == 0 or i ==1:
                    self.models.append(model_class(60, 64, 2, 7))
                else:
                    self.models.append(model_class(60, 32, 2, 7))
                    
                self.models[i].load_state_dict(combined_state_dict[f'model_{i+1}'])

    def forward(self, x, lengths):
        
        outputs = [model(x, lengths) for model in self.models]
        return torch.mean(torch.stack(outputs), dim=0)

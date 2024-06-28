from torch.utils.data import Dataset
import torch
from transforms import *

class CustomDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = [torch.tensor(x, dtype=torch.float) for x in X]
        self.y = torch.tensor(y, dtype=torch.float)
        self.max_len = 150
        self.transform = transform
        
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]

        if self.transform:
            x = self.transform(x)

        length = len(x[0])
        padded_x = torch.zeros((60, self.max_len), dtype=torch.float)
        sequence_length = x.shape[1]
        padded_x[:, :sequence_length] = x
        return padded_x, y, length
        # return x, y, length
    
class Test_Dataset(Dataset):
    def __init__(self, X):
        self.X = X
        self.max_len = 1500

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = self.X[idx]
        subject_id = data['subject_id']
        date = data['date'].strftime('%Y-%m-%d')
        sequence = data['sequence']

        sequence_tensor = torch.tensor(sequence, dtype=torch.float)
        
        length = len(sequence_tensor[0])
        padded_x = torch.zeros((60, self.max_len), dtype=torch.float)
        sequence_length = sequence_tensor.shape[1]
        padded_x[:, :sequence_length] = sequence_tensor
        
        return subject_id, date, padded_x, length
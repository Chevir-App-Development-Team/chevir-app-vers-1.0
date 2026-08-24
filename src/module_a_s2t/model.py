import torch
import torch.nn as nn

class SignLanguageLSTM(nn.Module):
    def __init__(self, input_size=1662, hidden_size=128, num_layers=2, num_classes=3):
        """
        Lightweight LSTM model to process sequence of keypoints.
        input_size: 1662 (default flattened MediaPipe Holistic array)
        """
        super(SignLanguageLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        out, (hn, cn) = self.lstm(x)
        
        # We only need the output of the last time step for classification
        out = self.fc(out[:, -1, :])
        return out

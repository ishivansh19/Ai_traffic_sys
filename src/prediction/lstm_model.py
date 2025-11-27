import torch
import torch.nn as nn

class TrafficLSTM(nn.Module):
    """
    Long Short-Term Memory (LSTM) Network for Traffic Forecasting.
    Input: Sequence of past traffic states (e.g., last 10 minutes).
    Output: Prediction of future traffic state (e.g., next 5 minutes).
    """
    def __init__(self, input_size=3, hidden_size=64, num_layers=2, output_size=1):
        super(TrafficLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        # batch_first=True means input shape is (batch, seq_len, features)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully Connected Layer to map LSTM output to prediction
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
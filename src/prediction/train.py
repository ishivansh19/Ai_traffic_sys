import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import pickle  # Added for saving the scaler
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(src_dir)

# Import the model architecture
from prediction.lstm_model import TrafficLSTM

def create_sequences(data, seq_length, prediction_horizon):
    xs, ys = [], []
    for i in range(len(data) - seq_length - prediction_horizon):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length + prediction_horizon, 1] # Index 1 is 'queue_length'
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    # 1. Configuration
    SEQ_LENGTH = 10
    PREDICT_AHEAD = 1
    BATCH_SIZE = 64
    EPOCHS = 50
    LEARNING_RATE = 0.001
    
    # Paths
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
    data_path = os.path.join(root_dir, 'data', 'traffic_history.csv')
    model_save_path = os.path.join(root_dir, 'models', 'lstm', 'traffic_model.pth')
    scaler_save_path = os.path.join(root_dir, 'models', 'lstm', 'traffic_scaler.pkl')
    
    print(f"--- TRAINING STARTED ---")
    print(f"Loading data from: {data_path}")
    
    # 2. Load and Preprocess Data
    df = pd.read_csv(data_path)
    feature_cols = ['total_vehicles', 'queue_length', 'network_speed', 'emergency_active']
    data = df[feature_cols].values.astype(float)
    
    # Normalize data
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data)
    
    # --- SAVE THE SCALER ---
    # This is crucial for live inference later
    if not os.path.exists(os.path.dirname(scaler_save_path)):
        os.makedirs(os.path.dirname(scaler_save_path))
        
    with open(scaler_save_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_save_path}")
    
    # Create Sequences
    X, y = create_sequences(data_normalized, SEQ_LENGTH, PREDICT_AHEAD)
    
    # Split into Train/Test
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Convert to Tensors
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)
    
    # DataLoaders
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 3. Initialize Model
    model = TrafficLSTM(input_size=4, hidden_size=64, num_layers=2, output_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 4. Training Loop
    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}")

    # 5. Evaluation
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs.squeeze(), y_test)
        print(f"Final Test Loss (MSE): {test_loss.item():.6f}")
        
    # 6. Save Model
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to: {model_save_path}")

if __name__ == "__main__":
    train_model()
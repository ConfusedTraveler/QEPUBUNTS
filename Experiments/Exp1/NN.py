import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np

def get_data(path, length):
    df = pd.read_csv(path)
    df = df.iloc[:length, :]
    return df.iloc[:, 0].values 

# 定义简单的前馈神经网络
class SimpleNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=16, output_size=1):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def create_dataset(series):
    X = series[:-1].reshape(-1, 1)
    y = series[1:].reshape(-1, 1)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def model_train_pred(data, output_path, N, epochs=50, lr=0.01):
    # Split data
    train_size = int(len(data) * 0.8)
    train_series, test_series = data[:train_size], data[train_size:]
    
    # Create dataset
    X_train, y_train = create_dataset(train_series)
    X_test, y_test = create_dataset(test_series)
    
    # Define model
    model = SimpleNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    
    # Prediction
    model.eval()
    with torch.no_grad():
        predictions = model(X_test).numpy().flatten()
    
    # Save results
    with open(output_path, "a") as f:
        for j in range(len(predictions)):
            f.write(f"{N},{round(predictions[j])},{test_series[j+1]}\n")
    
    print("True and predicted values saved to ", output_path)
    return

if __name__ == "__main__":
    N_values = [5,10,15,20,25,30,35,40,45,50]
    n = 50000
    a_values = [1,15]

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)
    grandparent_directory = os.path.dirname(parent_directory)

    for a in a_values:
        for N in N_values:
            input_path = f"{grandparent_directory}/Datasets/Markov/Exp1/markov_ts_{N}_50000_{a}.csv"
            output_path = f"Results/NN/nn_pred_{N}_{n}_{a}.csv"
            series = get_data(input_path, n)
            model_train_pred(series, output_path, N)

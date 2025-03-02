import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import importlib
import utils
importlib.reload(utils)
from utils import get_universe_adjusted_series, scale_weights_to_one, backtest, metrics_from_holdings
from tqdm import tqdm

# Data normalization: standardize each feature globally
def normalize_features(arr):
    # arr shape: (T, N, F)
    mean = np.mean(arr, axis=(0,1), keepdims=True)
    std = np.std(arr, axis=(0,1), keepdims=True) + 1e-6
    return (arr - mean) / std

data_dir = "/kaggle/input/qrt-quant-dataquest-2025-iit-delhi/Learning"
features = pd.read_parquet(os.path.join(data_dir, "features.parquet"))
universe = pd.read_parquet(os.path.join(data_dir, "universe.parquet"))
returns = pd.read_parquet(os.path.join(data_dir, "returns.parquet"))

feature_names = [f"f{i}" for i in range(1, 26)]
features_array = np.stack([features[f].values for f in feature_names], axis=-1)
returns_array = returns.values

features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
returns_array = np.nan_to_num(returns_array, nan=0.0, posinf=0.0, neginf=0.0)
features_array = normalize_features(features_array)

# Enhanced model architecture with two hidden layers, BatchNorm and Dropout
class StockMLP(nn.Module):
    def __init__(self, input_dim=25, hidden_dim=64, dropout_rate=0.2):
        super(StockMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.bn2 = nn.BatchNorm1d(hidden_dim//2)
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_dim//2, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x))) 
        x = F.relu(self.bn2(self.fc2(x))) 
        x = self.dropout(x)
        return self.out(x).squeeze()

def train_model(features_array, returns_array, num_epochs=100, lr=1e-3, weight_decay=1e-4):
    T, N, F = features_array.shape
    model = StockMLP(input_dim=F, hidden_dim=64, dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    
    features_tensor = torch.tensor(features_array, dtype=torch.float32).contiguous()
    returns_tensor = torch.tensor(returns_array, dtype=torch.float32).contiguous()
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        x = features_tensor.reshape(-1, F)
        y = returns_tensor.reshape(-1)
        pred_scores = model(x).view(-1)
        if torch.isnan(pred_scores).any() or torch.isnan(y).any():
            continue
        std_y_pred = torch.std(y * pred_scores) + 1e-6
        if std_y_pred.item() == 0:
            continue
        loss = -torch.mean(y * pred_scores) / std_y_pred
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    return model

model = train_model(features_array, returns_array)

def get_weights(features: pd.DataFrame, today_universe: pd.Series, prev_weights: dict = None) -> dict[str, float]:
    if features.empty:
        return today_universe[today_universe==1].replace(0, np.nan).dropna().to_dict()
    latest_features = np.stack([features[f].iloc[-1].values for f in feature_names], axis=-1)
    latest_features = torch.tensor(latest_features, dtype=torch.float32)
    scores = model(latest_features).detach().numpy().flatten()
    score_series = pd.Series(scores, index=today_universe.index).astype(float)
    # Only consider tradable stocks
    tradable_scores = score_series[today_universe == 1]
    tradable_scores_tensor = torch.tensor(tradable_scores.values, dtype=torch.float32)
    # Differentiable projection using softmax approximations
    L = F.softmax(tradable_scores_tensor, dim=0)
    S = F.softmax(-tradable_scores_tensor, dim=0)
    raw_weights = 0.5 * L - 0.5 * S
    norm_factor = torch.sum(torch.abs(raw_weights)) + 1e-6
    weights_tensor = raw_weights / norm_factor
    weights = weights_tensor.numpy().flatten()
    alpha = pd.Series(weights, index=tradable_scores.index).astype(float)
    if prev_weights is not None:
        for stock in prev_weights.keys():
            if stock in alpha.index and today_universe.loc[stock] == 0:
                alpha.loc[stock] = prev_weights[stock]
    # Ensure unit capital
    alpha = alpha / alpha.abs().sum()
    return alpha.replace(0, np.nan).dropna().to_dict()

prev_weights = None
positions_dict = {}
selected_dates = features.index[(features.index >= "2005-01-03") & (features.index <= "2014-12-31")]

for day in tqdm(selected_dates, desc="Processing Days"):
    current_features = features.loc[:day]
    current_universe = universe.loc[day]
    weights = get_weights(current_features, current_universe, prev_weights)
    positions_dict[day] = weights
    prev_weights = weights

positions_df = pd.DataFrame(positions_dict).T
nsr = metrics_from_holdings(positions_df, returns, universe)
positions_df.to_csv("submission.csv")
print("NSR:", nsr)
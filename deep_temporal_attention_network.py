import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

# Set deterministic seed for reproducibility
torch.manual_seed(42)

class MultiFactorTemporalTransformer(nn.Module):
    """
    State-of-the-Art Deep Learning Encoder for Financial Time Series.
    Combines Multi-Head Self-Attention with Residual Feed-Forward networks
    to capture dynamic market regimes and nonlinear factor interactions.
    """
    def __init__(self, num_features: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super(MultiFactorTemporalTransformer, self).__init__()
        
        # Feature Projection Layer
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Positional Encoding Layer for sequential order
        self.pos_encoder = nn.Parameter(torch.zeros(1, 100, d_model))
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output Heads: Dual-Task Prediction (Return Alpha + Realized Volatility)
        self.alpha_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Predicted 5-day return score
        )
        self.volatility_head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Softplus()     # Guaranteed positive volatility estimate
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, sequence_length, num_features]
        batch_size, seq_len, _ = x.shape
        
        # Project raw features to embedding dimension
        out = self.input_projection(x) + self.pos_encoder[:, :seq_len, :]
        
        # Pass through Transformer layers
        transformed = self.transformer_encoder(out)
        
        # Pool across sequence length (use last time step encoding)
        last_step = transformed[:, -1, :]
        
        alpha_score = self.alpha_head(last_step)
        predicted_vol = self.volatility_head(last_step)
        
        return alpha_score, predicted_vol

# Example Workflow & Execution Harness
if __name__ == "__main__":
    # Hyperparameters
    BATCH_SIZE = 32
    SEQ_LEN = 30  # 30 trading days window
    NUM_FEATURES = 10 # Price momentum, RSI, Volatility, Order Imbalance, Sentiment score, etc.
    
    print("[+] Initializing Multi-Factor Temporal Transformer Model...")
    model = MultiFactorTemporalTransformer(num_features=NUM_FEATURES, d_model=64, nhead=4)
    
    # Synthetic batch representing 32 stock sequences over 30 days
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LEN, NUM_FEATURES)
    
    # Forward Pass
    alpha_predictions, vol_predictions = model(dummy_input)
    
    print(f"[+] Output Shapes -> Alpha Scores: {alpha_predictions.shape}, Volatility: {vol_predictions.shape}")
    print(f"Sample Top Alpha Score: {alpha_predictions[0].item():.4f}")
    print(f"Sample Predicted Volatility: {vol_predictions[0].item():.4f}")

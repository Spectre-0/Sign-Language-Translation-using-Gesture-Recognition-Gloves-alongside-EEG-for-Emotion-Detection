import torch.nn as nn

# Model Definition (CNN + Transformer)
class SignLanguageModel(nn.Module):
    def __init__(self, input_channels=14, num_labels=27):
        super(SignLanguageModel, self).__init__()
        
        self.norm = nn.BatchNorm1d(input_channels)

        self.conv_stack = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool1d(2),
        )
        
        # Transformer Encoder
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=256, dropout=0.3, nhead=8, dim_feedforward=256, activation='relu', batch_first=False)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3, enable_nested_tensor=False)
        
        # Final Classification Layer
        self.fc = nn.Linear(256, num_labels)
    
    def forward(self, x):
        # x shape: (batch_size, 19, 15)
        x = x.permute(0, 2, 1)  # (batch_size, 15, 19) -> (batch_size, input_channels, seq_length)
        x = self.norm(x)
        
        # CNN part
        x = self.conv_stack(x)
        
        # Transformer part
        x = x.permute(2, 0, 1)  # (batch_size, channels, seq_length) -> (seq_length, batch_size, channels)
        x = self.transformer(x)
        
        # Global Average Pooling
        x = x.mean(dim=0)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x
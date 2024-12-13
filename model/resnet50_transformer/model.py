
import torch
from torch import nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image
import math
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[num_tokens, batch_size, embedding_dim]``
        """

        seq_len = x.size(0)
        return self.dropout(x + self.pe[:seq_len])

class Resnet_50(nn.Module):
    def __init__(self, freeze=False):
        super(Resnet_50, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Keep up to the second-last layer (before pooling)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(2048, 256)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in list(self.model.parameters())[-15:]:
                param.requires_grad = True

    def forward(self, x, training=True):

        features = self.model(x)

        pooled_features = self.global_avg_pool(features) 
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        output = self.fc(pooled_features)

        return output


class LipReadingFactorized(nn.Module):
    def __init__ (self):
        super(ViTLipReadingFactorized, self).__init__()
        self.spatial_transformer_encoder = Resnet_50()
        self.pos_encoding = PositionalEncoding(d_model = 256, dropout=0.3)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=4, 
            dropout=0.2,
            activation='relu',
            batch_first=True
        )

        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

        self.final_layer = nn.Sequential(nn.Linear(256, 301),
                                         nn.BatchNorm1d(301),
                                         nn.LeakyReLU(0.2),
                                         nn.Dropout(0.3),
                                         nn.Linear(301, 301))

    def forward(self, x, training=True):
        BS, NF, NC, H, W = x.shape

        # by frames, instead of batchs
        x = x.permute(1,0,2,3,4)

        # Spatial Transformer Encoder
        frames = []
        for frame in x:
            # Get CLS token from each frame using ViT-b32
            frame = self.spatial_transformer_encoder(frame, training)
            frames.append(frame)
    
        frames = torch.stack(frames, dim=0)  # Shape: (NF, BS, D_MODEL)
        
        # Average CLS tokens across frames
        cls_token = torch.mean(frames, dim=0) 
        frames = torch.concat([cls_token.unsqueeze(0), frames])

        # Add positional encoding

        x = self.pos_encoding(frames).permute(1,0,2)

        # Transformer Encoder (for temporal processing)
        x = self.transformer_encoder(x)[:, 0, :]  # We take the output from the first (CLS) token
        
        # Final Classification
        x = self.final_layer(x)

        return x

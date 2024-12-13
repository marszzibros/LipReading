
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
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN,self).__init__()
        # First we'll define our layers
        self.conv1 = nn.Conv2d(3,32,kernel_size=5,padding=2)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,32,kernel_size=5,padding=2)
        self.batchnorm2 = nn.BatchNorm2d(32)
        self.maxpool2 = nn.MaxPool2d(2,2)
        self.linear = nn.Linear(512, 256)


    
    def forward(self,x):
        BS, C, W, H = x.shape
        x = F.relu(self.conv1(x))
        x = self.batchnorm1(x)
        x = self.maxpool1(x)
        x = F.relu(self.conv2(x))
        x = self.batchnorm2(x)
        x = self.maxpool2(x)

        x = self.linear(x.reshape(BS, 32 * 4 * 4))

        return x

class Resnet_101(nn.Module):
    def __init__(self, freeze=False):
        super(Resnet_101, self).__init__()
        self.model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1) 
        self.model = nn.Sequential(*list(self.model.children())[:-2])  # Keep up to the second-last layer (before pooling)


        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Linear(2048, 256)

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            for param in list(self.model.parameters())[-35:]:
                param.requires_grad = True

    def forward(self, x, training=True):

        features = self.model(x)

        pooled_features = self.global_avg_pool(features) 
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        output = self.fc(pooled_features)

        return output

class ViT_b_32(nn.Module):
    def __init__(self, freeze=False):
        super(ViT_b_32, self).__init__()

        # Load pre-trained ViT model
        self.model = models.vit_b_32(weights="ViT_B_32_Weights.IMAGENET1K_V1") 
        self.cnn = CNN()

        if not freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            # Unfreeze the last few layers for fine-tuning
            for param in self.model.encoder.layers[-6:].parameters():
                param.requires_grad = True

    def forward(self, x, training=True):
        BS = x.shape[0]
        
        # Process input image into patch embeddings
        x = self.model._process_input(x).transpose(1,0)
        frames = []
        for frame in x:
        
            frames.append(self.cnn(frame.view(BS, 3, 16, 16)))
            
        x = torch.stack(frames, dim=0).transpose(1,0)


        # Expand CLS token to batch size
        cls_token = self.model.class_token.expand(BS, -1, -1)
        x = torch.cat((cls_token, x), dim=1)


        
        # Add positional embeddings
        x = x + self.model.encoder.pos_embedding.expand(BS, -1, -1)
        
        if training:
            x = self.model.encoder.dropout(x)
        
        # Transformer Encoder 
        x = self.model.encoder.layers(x)

        cls_output = x[:, 0]

        return cls_output



class ViTLipReadingFactorized(nn.Module):
    def __init__ (self):
        super(ViTLipReadingFactorized, self).__init__()
        self.spatial_transformer_encoder = Resnet_101()
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

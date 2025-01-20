import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl

# -------------------------------
# Cola Model Definition
# -------------------------------

class DotProduct(nn.Module):
    """Normalized dot product similarity."""
    def forward(self, anchor, positive):
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        return torch.matmul(anchor, positive.T)


class BilinearProduct(nn.Module):
    """Bilinear similarity with a trainable weight matrix."""
    def __init__(self, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(dim, dim))
    
    def forward(self, anchor, positive):
        projection_positive = torch.matmul(self.weight, positive.T)
        return torch.matmul(anchor, projection_positive)


class Encoder(nn.Module):
    def __init__(self, drop_connect_rate=0.1):
        super(Encoder, self).__init__()
        self.cnn1 = nn.Conv2d(1, 3, kernel_size=3)
        self.efficientnet = EfficientNet.from_name(
            "efficientnet-b0", include_top=False, drop_connect_rate=drop_connect_rate
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = self.cnn1(x)
        x = self.efficientnet(x)
        return x.squeeze(3).squeeze(2)  # Flatten spatial dimensions


class Cola(pl.LightningModule):
    def __init__(self, embedding_dim=512, p=0.1, similarity_type="dot", temperature=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        self.encoder = Encoder(drop_connect_rate=p)
        self.g = nn.Linear(1280, embedding_dim)  # EfficientNet output is 1280-dim
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        self.linear = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.dropout = nn.Dropout(p=p)

        # Similarity layer
        if similarity_type == "bilinear":
            self.similarity = BilinearProduct(embedding_dim)
        else:
            self.similarity = DotProduct()
        
        self.similarity_type = similarity_type
        self.temperature = temperature

    def forward(self, x):
        x1, x2 = x

        x1 = self.dropout(self.encoder(x1))
        x1 = self.dropout(self.g(x1))
        x1 = self.dropout(torch.tanh(self.layer_norm(x1)))

        x2 = self.dropout(self.encoder(x2))
        x2 = self.dropout(self.g(x2))
        x2 = self.dropout(torch.tanh(self.layer_norm(x2)))

        return x1, x2

    def compute_similarity(self, x1, x2):
        similarities = self.similarity(x1, x2)
        if self.similarity_type == "dot":
            similarities = similarities / self.temperature
        return similarities

    def training_step(self, batch, batch_idx):
        x1, x2 = self(batch)

        # Compute similarity logits
        y = torch.arange(x1.size(0), device=x1.device)
        y_hat = self.compute_similarity(x1, x2)

        # Compute loss and accuracy
        loss = F.cross_entropy(y_hat, y)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2 = self(batch)
        y = torch.arange(x1.size(0), device=x1.device)
        y_hat = self.compute_similarity(x1, x2)

        loss = F.cross_entropy(y_hat, y)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x1, x2 = self(batch)
        y = torch.arange(x1.size(0), device=x1.device)
        y_hat = self.compute_similarity(x1, x2)

        loss = F.cross_entropy(y_hat, y)
        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

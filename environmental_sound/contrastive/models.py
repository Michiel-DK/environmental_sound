import torch
import torch.nn.functional as F
from torch import nn
from efficientnet_pytorch import EfficientNet
import pytorch_lightning as pl
from environmental_sound.contrastive.model_utils import DotProduct, BilinearProduct


# -------------------------------
# Models
# -------------------------------

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
    

class SimCLRFineTuner(pl.LightningModule):
    def __init__(self, encoder, embedding_dim=512, temperature=0.1, classes=50):
        super().__init__()
        self.encoder = encoder  # Pretrained Cola model
        self.projection_head = nn.Sequential(
            nn.Linear(1280, 512),  # Match the encoder's output dimension (1280) to the projection head's input
            nn.ReLU(),
            nn.Linear(512, embedding_dim)  # Project down to the embedding_dim (512)
        )
        self.temperature = temperature
        self.classes = classes

    @classmethod
    def from_pretrained(cls, checkpoint_path, encoder, **kwargs):
        """
        Load a SimCLRFineTuner model from a checkpoint with a provided encoder.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file.
            encoder (nn.Module): Pretrained encoder to use.
            **kwargs: Additional arguments to pass to the model.
        """
        return cls.load_from_checkpoint(
            checkpoint_path,
            encoder=encoder,
            **kwargs
        )

    def compute_similarity(self, x1, x2):
        # Normalize embeddings
        x1 = F.normalize(x1, dim=-1)
        x2 = F.normalize(x2, dim=-1)
        return torch.matmul(x1, x2.T)

    def contrastive_loss(self, x1, x2, labels):
        # Compute similarity scores
        similarities = self.compute_similarity(x1, x2) / self.temperature  # Shape: (batch_size, batch_size)

        # Create the target tensor for contrastive loss
        batch_size = x1.size(0)
        targets = torch.arange(batch_size, device=similarities.device)

        # Compute cross-entropy loss
        loss = F.cross_entropy(similarities, targets)

        # Compute accuracy
        _, predicted = torch.max(similarities, dim=1)
        acc = (predicted == targets).float().mean()

        return loss, acc

    def training_step(self, batch, batch_idx):
        x1, x2, labels = batch  # Unpack batch
        x1 = self.encoder(x1)  # Process x1 through the encoder
        x2 = self.encoder(x2)  # Process x2 through the encoder
        x1 = self.projection_head(x1)
        x2 = self.projection_head(x2)

        # Calculate loss and accuracy
        loss, acc = self.contrastive_loss(x1, x2, labels)

        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x1, x2, labels = batch  # Unpack batch
        x1 = self.encoder(x1)  # Process x1 through the encoder
        x2 = self.encoder(x2)  # Process x2 through the encoder
        x1 = self.projection_head(x1)
        x2 = self.projection_head(x2)

        # Calculate loss and accuracy
        loss, acc = self.contrastive_loss(x1, x2, labels)

        # Log validation metrics
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x1, x2, labels = batch  # Unpack batch
        x1 = self.encoder(x1)  # Process x1 through the encoder
        x2 = self.encoder(x2)  # Process x2 through the encoder
        x1 = self.projection_head(x1)
        x2 = self.projection_head(x2)

        # Calculate loss and accuracy
        loss, acc = self.contrastive_loss(x1, x2, labels)

        # Log test metrics
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)



class AudioClassifier(pl.LightningModule):
    def __init__(self, encoder, classes=50, embedding_dim=512, p=0.1, freeze_encoder=False, lr_encoder=1e-5, lr_downstream=1e-4):
        super().__init__()
        self.save_hyperparameters()
        
        self.lr_encoder = lr_encoder
        self.lr_downstream = lr_downstream
        
        self.p = p
        self.embedding_dim = embedding_dim

        self.dropout = nn.Dropout(p=self.p)

        # Encoder
        self.encoder = encoder
        
        self.freeze_encoder = freeze_encoder
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False  # Freeze encoder for linear probing

        # MLP for classification
        self.g = nn.Linear(1280, embedding_dim)
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)

        self.fc1 = nn.Linear(embedding_dim, 256)
        self.fc2 = nn.Linear(256, classes)

    def forward(self, x):
        # Pass through encoder
        x = self.dropout(self.encoder(x))

        # Intermediate layers
        x = self.dropout(self.g(x))
        x = self.dropout(torch.tanh(self.layer_norm(x)))

        # Classification head
        x = F.relu(self.dropout(self.fc1(x)))
        y_hat = self.fc2(x)

        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).float().mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).float().mean()

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x)

        loss = F.cross_entropy(y_hat, y)

        _, predicted = torch.max(y_hat, 1)
        acc = (predicted == y).float().mean()

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        # Configure different learning rates for encoder and classification head
        if self.freeze_encoder:
            # Only optimize the classification head
            optimizer = torch.optim.Adam([
                {'params': self.g.parameters(), 'lr': self.lr_downstream},
                {'params': self.layer_norm.parameters(), 'lr': self.lr_downstream},
                {'params': self.fc1.parameters(), 'lr': self.lr_downstream},
                {'params': self.fc2.parameters(), 'lr': self.lr_downstream},
            ])
        else:
            # Optimize both the encoder and classification head
            optimizer = torch.optim.Adam([
                {'params': self.encoder.parameters(), 'lr': self.lr_encoder},
                {'params': self.g.parameters(), 'lr': self.lr_downstream},
                {'params': self.layer_norm.parameters(), 'lr': self.lr_downstream},
                {'params': self.fc1.parameters(), 'lr': self.lr_downstream},
                {'params': self.fc2.parameters(), 'lr': self.lr_downstream},
            ])

        return optimizer

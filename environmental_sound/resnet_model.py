import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchmetrics

class CustomCNNLightning(pl.LightningModule):
    def __init__(self, num_classes=50, lr=1e-4, input_shape=(1, 13, 173)):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr

        # Define CNN architecture
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # Removed additional pooling to avoid zero-size outputs
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self._compute_flattened_size(input_shape)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

        # Initialize TorchMetrics accuracy metrics with task and num_classes
        self.train_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=self.hparams.num_classes)

    def _compute_flattened_size(self, input_shape):
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            features = self.cnn(dummy_input)
            pooled_features = self.adaptive_pool(features)
            self.flattened_size = pooled_features.view(1, -1).size(1)

    def forward(self, x):
        # Ensure input has a channel dimension
        if x.dim() == 3:
            x = x.unsqueeze(1)  # shape: [batch, 1, height, width]
        features = self.cnn(x)
        pooled_features = self.adaptive_pool(features)
        logits = self.classifier(pooled_features)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        target = torch.argmax(y, dim=1)
        loss = F.cross_entropy(logits, target)
        
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, target)
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        target = torch.argmax(y, dim=1)
        loss = F.cross_entropy(logits, target)
        
        preds = torch.argmax(logits, dim=1)
            
        self.val_accuracy.update(preds, target)
        
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_accuracy, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        target = torch.argmax(y, dim=1)
        loss = F.cross_entropy(logits, target)
        
        preds = torch.argmax(logits, dim=1)
        self.test_accuracy.update(preds, target)
        
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', self.test_accuracy, prog_bar=True)

    def on_train_epoch_end(self):
        train_loss = self.trainer.callback_metrics.get('train_loss')
        train_acc = self.trainer.callback_metrics.get('train_acc')
        print(f"Epoch {self.current_epoch}: Train Loss = {train_loss:.4f}, Train Acc = {train_acc:.4f}")
        self.train_accuracy.reset()

    def on_validation_epoch_end(self):
        val_loss = self.trainer.callback_metrics.get('val_loss')
        val_acc = self.trainer.callback_metrics.get('val_acc')
        
        val_acc_value = self.val_accuracy.compute()
        print(f"Epoch {self.current_epoch}: Val Loss = {val_loss:.4f}, Val Acc (manual) = {val_acc_value:.4f}")
        
        #print(f"Epoch {self.current_epoch}: Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.4f}")
        self.val_accuracy.reset()

    def on_test_epoch_end(self):
        test_loss = self.trainer.callback_metrics.get('test_loss')
        test_acc = self.trainer.callback_metrics.get('test_acc')
        print(f"Test Results: Loss = {test_loss:.4f}, Acc = {test_acc:.4f}")
        self.test_accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.1,
                patience=3,
                verbose=True
            ),
            'monitor': 'val_loss'
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

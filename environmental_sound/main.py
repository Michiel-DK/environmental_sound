import pandas as pd

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from environmental_sound.dataloader import MFCCDataModule
from environmental_sound.resnet_model import CustomCNNLightning
from environmental_sound.transformations import RandomAudio, TimeStretch, MelSpectrogram, SpecAugment, SpectToImage

import albumentations



def main_run():

    # Initialize WandB logger
    wandb_logger = WandbLogger(project='environmental-sound', group="base_resnet50", name='base_resnet50_aug')

    # EarlyStopping callback: stops training if no improvement in 'val_loss' for 3 epochs
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=13,
        verbose=True,
        mode='min'
    )

    # LearningRateMonitor: logs learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ModelCheckpoint: saves top 3 models based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='resnet50-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'
    )

    # Determine accelerator: MPS, CUDA, or CPU
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    # Initialize Trainer with WandB logger, callbacks, and accelerator configuration
    trainer = Trainer(
        max_epochs=100,
        accelerator=accelerator,
        devices=1,  # Use one device; adjust as needed
        logger=wandb_logger,
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback]
    )

    labels_df = pd.read_csv('audio_data/esc50.csv')

    train_transform = albumentations.Compose([
        RandomAudio(seconds=1, p=0.5),    # <-- augmentation
        TimeStretch(p=0.8),              # <-- augmentation
        #MelSpectrogram(parameters={"n_mels": 128, "fmax": 8000}, p=1.0),  # <-- necessary
        SpecAugment(p=0.8),              # <-- augmentation
        SpectToImage(p=1.0)              # <-- can be seen as final step
    ])

    val_test_transform = albumentations.Compose([
        # No random augmentation here;
        # Just do the minimal steps needed to get the final representation
       # MelSpectrogram(parameters={"n_mels": 128, "fmax": 8000}, p=1.0),
        SpectToImage(p=1.0)
    ])

    # Initialize the data module
    data_module = MFCCDataModule(df=labels_df, target_size=(13, 173), audio_transform_train=train_transform, audio_transform_val_test=val_test_transform, batch_size=32)
    
    #data_module.setup()

    # Initialize the ResNet50 Lightning model
    model = CustomCNNLightning(num_classes=50, lr=1e-3, input_shape=(1, 13, 173))

    # Train the model using the data module
    trainer.fit(model, datamodule=data_module)

    # Evaluate on the test set
    trainer.test(model, datamodule=data_module)
    
if __name__ == '__main__':
    main_run()
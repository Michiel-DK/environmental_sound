import pandas as pd

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from environmental_sound.data.dataloader import MFCCDataModule
from environmental_sound.supervised.resnet_model import CustomCNNLightning
from environmental_sound.supervised.transformations import RandomAudio, TimeStretch, MelSpectrogram, SpecAugment, SpectToImage

import albumentations

from pytorch_lightning.loggers import TensorBoardLogger

import hydra
from omegaconf import DictConfig, OmegaConf

from bunch import Bunch

import os


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main_run(cfg: DictConfig):
    
    #access hydra config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    #access config values with Bunch for easier implementation
    project_config = Bunch(config_dict['wandb'])
    trainer_config = Bunch(config_dict['trainer'])
        
    # Initialize WandB logger
    if trainer_config.wandb_log:
        wandb_logger = WandbLogger(project=project_config.project\
            ,group=project_config.group, name=project_config.name\
                ,config=config_dict['trainer'])
    else:
        tensor_logger = TensorBoardLogger(save_dir="tensorboard_logs/", name=project_config.name)
    
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
        logger=[wandb_logger if trainer_config.wandb_log else tensor_logger],
        callbacks=[early_stop_callback, lr_monitor, checkpoint_callback]
    )

    csv_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio_data/esc50.csv')
    labels_df = pd.read_csv(csv_path)

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
    
    
    if trainer_config.wandb_log:
        run = wandb_logger.experiment
        run.config.update(
            {"train_transform": train_transform, 
             "val_test_transform": val_test_transform,
             "dataset_size": len(labels_df),
             }
        )
        
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'audio_data/44100/')

    # Initialize the data module
    data_module = MFCCDataModule(df=labels_df, target_size=(trainer_config.n_mfcc, 173), audio_transform_train=train_transform\
        ,audio_transform_val_test=val_test_transform, batch_size=trainer_config.batch_size \
        ,sample_subset=trainer_config.sample_subset, train_pct=trainer_config.train_size, val_pct=trainer_config.val_size, test_pct=trainer_config.test_size\
            ,data_path=data_path)
    
    # Initialize the ResNet50 Lightning model
    model = CustomCNNLightning(num_classes=trainer_config.num_classes, lr=trainer_config.learning_rate, input_shape=(1, trainer_config.n_mfcc, 173))

    # Train the model using the data module
    trainer.fit(model, datamodule=data_module)

    # Evaluate on the test set
    trainer.test(model, datamodule=data_module)
    
if __name__ == '__main__':
    main_run()
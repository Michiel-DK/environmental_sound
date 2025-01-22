import hydra
from omegaconf import DictConfig, OmegaConf

from bunch import Bunch

from glob import glob
import os
import torch

from sklearn.model_selection import train_test_split
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
import pytorch_lightning as pl


from environmental_sound.contrastive.datasets import ContrastiveAudioDatasetUnsupervised
from environmental_sound.utils.gcp import check_and_setup_directory
from environmental_sound.contrastive.model_utils import ReduceLROnPlateauCallback
from environmental_sound.contrastive.models import Cola



@hydra.main(config_path="conf", config_name="config", version_base=None)
def main_run(cfg: DictConfig):
    
    #access hydra config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    #access config values with Bunch for easier implementation
    project_config = Bunch(config_dict['wandb'])
    trainer_config = Bunch(config_dict['trainer_contrastive'])
    
    
    root_path = os.path.dirname(os.path.dirname(__file__))
    data_path = f'audio_data/{project_config.local_npy_dir}/'
        
    output_data_path = os.path.join(root_path, data_path)
    
    check_and_setup_directory(root_path, output_data_path, project_config.bucket_name, project_config.tar_blob_name)

    files = os.listdir(output_data_path)
        
    #filter for later finetuning
    filtered_files = [file for file in files if not any(file.startswith(prefix) for prefix in trainer_config.finetune_prefix)]
    
    files_paths = [os.path.join(output_data_path, f) for f in filtered_files]

    _train, test = train_test_split(files_paths, test_size=trainer_config.test_size, random_state=trainer_config.random_state)

    train, val = train_test_split(_train, test_size=trainer_config.val_size, random_state=trainer_config.random_state)

    train_data = ContrastiveAudioDatasetUnsupervised(train, augment=True, seg_length=trainer_config.seg_length, crop_size=trainer_config.crop_size)
    test_data = ContrastiveAudioDatasetUnsupervised(test, augment=False, seg_length=trainer_config.seg_length, crop_size=trainer_config.crop_size)
    val_data = ContrastiveAudioDatasetUnsupervised(val, augment=False, seg_length=trainer_config.seg_length, crop_size=trainer_config.crop_size)

    train_loader = DataLoader(
        train_data, batch_size=trainer_config.batch_size, num_workers=2, shuffle=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_data, batch_size=trainer_config.batch_size, num_workers=2, shuffle=False, persistent_workers=True
    )
    test_loader = DataLoader(
        test_data, batch_size=trainer_config.batch_size, shuffle=False, num_workers=2, persistent_workers=True
    )

    model = Cola(embedding_dim=trainer_config.embedding_dim, similarity_type=trainer_config.similarity_type, temperature=trainer_config.temperature)
    
    # Initialize WandB logger
    if trainer_config.wandb_log:
        wandb_logger = WandbLogger(project=project_config.project\
            ,group=project_config.group, name=project_config.name\
                ,config=config_dict['trainer_contrastive'])
    else:
        tensor_logger = TensorBoardLogger(save_dir="tensorboard_logs/", name=project_config.name)

    # EarlyStopping callback: stops training if no improvement in 'val_loss' for 3 epochs
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=21,
        verbose=True,
        mode='min'
    )

    # LearningRateMonitor: logs learning rate changes
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='contrastive-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )
    
    # Determine accelerator: MPS, CUDA, or CPU
    if torch.backends.mps.is_available():
        accelerator = "mps"
    elif torch.cuda.is_available():
        accelerator = "gpu"
    else:
        accelerator = "cpu"

    trainer = pl.Trainer(
        max_epochs=trainer_config.epochs,
        accelerator=accelerator,
        devices=1,  # Use one device; adjust as needed
        log_every_n_steps=len(train)//trainer_config.batch_size,
        logger=[wandb_logger if trainer_config.wandb_log else tensor_logger],
        callbacks=[ReduceLROnPlateauCallback(), early_stop_callback, lr_monitor, checkpoint_callback],
    )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(dataloaders=test_loader)
    
if __name__ == "__main__":
    main_run()

import os
import hydra
from omegaconf import DictConfig, OmegaConf

from bunch import Bunch
from sklearn.model_selection import train_test_split
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import torch
from torch.utils.data import DataLoader



from environmental_sound.utils.gcp import check_and_setup_directory
from environmental_sound.contrastive.finetune import AudioDatasetSupervised, ReduceLROnPlateauCallback
from environmental_sound.contrastive.encoder import AudioClassifier


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main_run(cfg: DictConfig):
    
    #access hydra config
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    #access config values with Bunch for easier implementation
    project_config = Bunch(config_dict['wandb'])
    trainer_config = Bunch(config_dict['trainer_finetune'])
    
    root_path = os.path.dirname(os.path.dirname(__file__))
    data_path = 'audio_data/44100_npy/'
        
    output_data_path = os.path.join(root_path, data_path)
    
    check_and_setup_directory(root_path, output_data_path, project_config.bucket_name, project_config.tar_blob_name)

    files = os.listdir(output_data_path)
    
    #filter extract test set unsupervised part to add for finetuning
    filtered_files_test = [file for file in files if not file.startswith(trainer_config.finetune_prefix)]
    
    _train, test = train_test_split(filtered_files_test, test_size=trainer_config.test_size, random_state=trainer_config.random_state) 
        
    #filter on spared data
    filtered_files_prefix = [file for file in files if file.startswith(trainer_config.finetune_prefix)]
    
    filtered_files = sorted(test+filtered_files_prefix)
    
    filtered_wav = [x.replace('.npy', '.wav') for x in filtered_files]
    
    files_paths = [os.path.join(output_data_path, f) for f in filtered_files]
    
    labels_df = pd.read_csv(os.path.join(root_path, 'audio_data', 'esc50.csv')).sort_values(by='filename')
    
    labels_list = labels_df[labels_df['filename'].isin(filtered_wav)].target.to_list()
    
    samples = list(zip(files_paths, labels_list))
        
    _train, test = train_test_split(
            samples, test_size=trainer_config.test_size, random_state=trainer_config.random_state, stratify=[a[1] for a in samples]
        )

    train, val = train_test_split(
            _train, test_size=trainer_config.val_size, random_state=trainer_config.random_state, stratify=[a[1] for a in _train]
        )

    train_data = AudioDatasetSupervised(train, augment=False)
    test_data = AudioDatasetSupervised(test, augment=False)
    val_data = AudioDatasetSupervised(val, augment=False)

    train_loader = DataLoader(
            train_data, batch_size=trainer_config.batch_size, num_workers=2, shuffle=True,persistent_workers=True
        )
    val_loader = DataLoader(
            val_data, batch_size=trainer_config.batch_size, num_workers=2, shuffle=False,persistent_workers=True
        )
    test_loader = DataLoader(
            test_data, batch_size=trainer_config.batch_size, shuffle=False, num_workers=2,persistent_workers=True
        )

    model = AudioClassifier()
    
    if trainer_config.wandb_log is False:
        
        tensor_logger = TensorBoardLogger(save_dir="tensorboard_logs/", name=project_config.name)
    else:
        wandb_logger = WandbLogger(project=project_config.project\
            ,group=project_config.group, name=project_config.name\
                ,config=config_dict['trainer_finetune'])
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints/',
        filename='contr_fine-{epoch:02d}-{val_loss:.2f}',
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
        
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=13,
        verbose=True,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    if trainer_config.contrastive_checkpoint is not None:
            
        ckpt = torch.load(os.path.join(root_path, 'checkpoints', trainer_config.contrastive_checkpoint), map_location=accelerator)

        model.load_state_dict(ckpt["state_dict"], strict=False)

    trainer = pl.Trainer(
            max_epochs=trainer_config.epochs,
            log_every_n_steps=15,
            accelerator=accelerator,
            devices=1,  # Use one device; adjust as needed
            logger=[wandb_logger if trainer_config.wandb_log else tensor_logger],
            callbacks=[early_stop_callback, lr_monitor, checkpoint_callback, ReduceLROnPlateauCallback()]
                )
    trainer.fit(model, train_loader, val_loader)

    trainer.test(test_dataloaders=test_loader)
    
if __name__ == '__main__':
    main_run()

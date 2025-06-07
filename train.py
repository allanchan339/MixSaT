import pytorch_lightning as pl
import argparse
import os
import torch
from src.research.training import LitModel
from src.dataloading.data_loading import LitDataModule
from src.utils.utils import load_and_flatten_toml_config
from src.utils.common import (
    setup_gpu_config, setup_seeds, setup_logger, 
    create_model, setup_callbacks, get_base_trainer_params
)


parser = argparse.ArgumentParser(description="MixSaT Training Script. All parameters are loaded from the specified TOML configuration file.")
parser.add_argument('--cfg', type=str, default='cfg/train.toml',
                    help='Path to the TOML configuration file (default: cfg/train.toml). This file must contain all necessary parameters.')


def train_logic(args):
    """Main training logic."""
    # Handle fast dev run configuration
    if hasattr(args, 'fast_dev_run') and args.fast_dev_run:
        args.limit_train_batches = 1.0  # Ensure float
        args.limit_val_batches = 1.0    # Ensure float
        args.max_epochs = 10             # Ensure int
        args.devices = 1  # Use single device for fast dev run
        args.batch_size = int(134/3)  # Use small batch size for fast dev run
        args.strategy = 'auto'  # strategy for fast dev run


    # Setup configurations
    num_gpus = setup_gpu_config(args)
    setup_seeds(args)
    logger_to_use = setup_logger(args)
    
    # Create model and data module
    model = create_model(args)
    litModel = LitModel(model, args)
    dataModule = LitDataModule(args)
    
    # Setup callbacks
    callbacks_list = setup_callbacks(args)
    
    # Configure trainer for training
    trainer_params = get_base_trainer_params(args, callbacks_list, logger_to_use)
    trainer_params.update({
        'min_epochs': args.min_epochs,
        'max_epochs': args.max_epochs,
        'check_val_every_n_epoch': args.check_val_every_n_epoch,
        'limit_train_batches': args.limit_train_batches,
        'limit_val_batches': args.limit_val_batches,
        'fast_dev_run': hasattr(args, 'fast_dev_run') and args.fast_dev_run
    })

    trainer = pl.Trainer(**trainer_params)

    # Handle checkpoint path for resuming training
    ckpt_path_val = args.ckpt_path if hasattr(args, 'ckpt_path') else None
    ckpt_to_fit = ckpt_path_val if ckpt_path_val and os.path.exists(ckpt_path_val) else None
    
    # Train the model
    trainer.fit(model=litModel, datamodule=dataModule, ckpt_path=ckpt_to_fit)
    
    # Evaluate the best model
    ckpt_for_eval = 'best' if not ckpt_to_fit else ckpt_to_fit
    if ckpt_path_val and os.path.exists(ckpt_path_val) and ckpt_to_fit != ckpt_path_val:
        ckpt_for_eval = ckpt_path_val

        trainer_params.update({
            'devices': 1, 
            'strategy': 'auto'  # Use DDP strategy for fast dev run

        })

    trainer = pl.Trainer(**trainer_params)  # Reinitialize trainer for evaluation
    trainer.test(datamodule=dataModule, ckpt_path=ckpt_for_eval) 
    trainer.predict(datamodule=dataModule, ckpt_path=ckpt_for_eval)


if __name__ == '__main__':
    torch.set_float32_matmul_precision = 'medium'  # Set float32 matmul precision for better performance
    cli_args = parser.parse_args()
    flat_config_dict = load_and_flatten_toml_config(cli_args.cfg)
    args_namespace = argparse.Namespace(**flat_config_dict)
    train_logic(args_namespace)

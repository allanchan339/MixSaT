import pytorch_lightning as pl
import argparse
import os

from src.research.training import LitModel
from src.dataloading.data_loading import LitDataModule
from src.utils.utils import load_and_flatten_toml_config
from src.utils.common import (
    setup_gpu_config, setup_seeds, setup_logger, 
    create_model, setup_callbacks, get_base_trainer_params
)


parser = argparse.ArgumentParser(description="MixSaT Testing Script. All parameters are loaded from the specified TOML configuration file.")
parser.add_argument('--cfg', type=str, default='cfg/test.toml',
                    help='Path to the TOML configuration file (default: cfg/test.toml). This file must contain all necessary parameters.')


def test_logic(args):
    """Main testing logic."""
    # Handle fast dev run configuration
    if hasattr(args, 'fast_dev_run') and args.fast_dev_run:
        args.limit_train_batches = 1.0  # Ensure float
        args.limit_val_batches = 1.0    # Ensure float
        args.max_epochs = 1             # Ensure int

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
    
    # Configure trainer for testing only
    trainer_params = get_base_trainer_params(args, callbacks_list, logger_to_use)
    trainer_params.update({
        'limit_train_batches': 0.0,  # Disable training
        'limit_val_batches': 0.0,    # Disable validation
        'fast_dev_run': hasattr(args, 'fast_dev_run') and args.fast_dev_run
    })

    trainer = pl.Trainer(**trainer_params)

    # Validate checkpoint path
    ckpt_path_val = args.ckpt_path if hasattr(args, 'ckpt_path') else None
    if not ckpt_path_val or not os.path.exists(ckpt_path_val):
        raise ValueError("A valid ckpt_path must be provided in the TOML configuration for testing.")
    
    # Run testing and prediction
    trainer.test(model=litModel, datamodule=dataModule, ckpt_path=ckpt_path_val)
    trainer.predict(model=litModel, datamodule=dataModule, ckpt_path=ckpt_path_val)


if __name__ == '__main__':
    cli_args = parser.parse_args()
    flat_config_dict = load_and_flatten_toml_config(cli_args.cfg)
    args_namespace = argparse.Namespace(**flat_config_dict)
    test_logic(args_namespace)

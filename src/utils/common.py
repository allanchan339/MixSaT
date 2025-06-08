import pytorch_lightning as pl
import torch
import os
import numpy as np

from src.research.training import LitModel
from src.dataloading.data_loading import LitDataModule
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, ModelSummary, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from src.research.model_twins_1d_group_SE import TwinsSVT_1d_group_SE
from pytorch_lightning.loggers import WandbLogger
from src.research.callbacks import OutputManagementCallback


def setup_gpu_config(args):
    """Setup GPU configuration and worker counts based on args."""
    devices_val = args.devices
    num_gpus = 0
    devices_str = str(devices_val)  # Handles int, string, or list from TOML

    if ',' in devices_str:
        num_gpus = len(devices_str.split(','))
    elif devices_str == "-1":  # PL convention for all available GPUs
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    else:  # Handles single GPU index or count
        try:
            parsed_device_int = int(devices_str)
            if parsed_device_int == 0:
                num_gpus = 1
            elif parsed_device_int > 0:
                num_gpus = parsed_device_int
        except ValueError:
            num_gpus = 0

    if isinstance(devices_val, list):  # If TOML provided a list of ints
        num_gpus = len(devices_val)
    
    if num_gpus == 0 and args.accelerator == 'gpu':
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1  # Default to 1 if no CUDA but gpu set
    if num_gpus == 0:
        num_gpus = 1  # Avoid division by zero

    # Setup worker counts
    if hasattr(args, 'max_num_worker') and args.max_num_worker is not None:
        args.max_num_worker = int(args.max_num_worker)  # Use TOML value
    else:
        args.max_num_worker = os.cpu_count() // num_gpus if num_gpus > 0 else os.cpu_count()

    if hasattr(args, 'warmup') and args.warmup is not None:
        args.warmup = int(args.warmup)  # Use TOML value
    else:
        args.warmup = (1000 // num_gpus) if num_gpus > 0 else 1000

    if hasattr(args, 'max_iters') and args.max_iters is not None:
        args.max_iters = int(args.max_iters)  # Use TOML value
    else:
        args.max_iters = (1000 // num_gpus) if num_gpus > 0 else 1000

    return num_gpus


def setup_seeds(args):
    """Setup random seeds for reproducibility."""
    np.random.seed(args.seed)
    pl.seed_everything(args.seed, workers=True)


def setup_logger(args):
    """Setup logger based on configuration."""
    logger_to_use = True  # Default PL logger
    if hasattr(args, 'logger_type') and args.logger_type.lower() == 'wandb':
        # Use experiment_name if provided and not empty, otherwise let WandB generate random name
        experiment_name = getattr(args, 'experiment_name', None)
        if experiment_name == "" or experiment_name is None:
            experiment_name = None  # Let WandB generate random name
        
        logger_to_use = WandbLogger(name=experiment_name,
                                  project=args.project,
                                  entity=args.entity,
                                  config=vars(args))
    return logger_to_use


def create_model(args):
    """Create and return the model instance."""
    model = TwinsSVT_1d_group_SE(num_classes=args.num_classes, 
                                frames_size=args.framerate * args.window_size,
                                s1_next_dim=args.s1_next_dim,
                                s1_patch_size=args.s1_patch_size,
                                s1_local_patch_size=args.s1_local_patch_size,
                                s1_global_k=args.s1_global_k,
                                s1_depth=args.s1_depth,
                                s2_next_dim=args.s2_next_dim,
                                s2_patch_size=args.s2_patch_size,
                                s2_local_patch_size=args.s2_local_patch_size,
                                s2_global_k=args.s2_global_k,
                                s2_depth=args.s2_depth,
                                peg_kernel_size=args.peg_kernel_size,
                                dropout=args.dropout,
                                Post_norm=args.Post_norm)
    return model


def setup_callbacks(args):
    """Setup and return list of callbacks."""
    output_mgmt_callback = OutputManagementCallback(
        logger_type=args.logger_type if hasattr(args, 'logger_type') else 'none',
        output_dir=args.output_dir
    )
    
    callbacks_list = [
        ModelCheckpoint(monitor='Valid/mAP', mode='max'), 
        ModelSummary(max_depth=-1),
        LearningRateMonitor(),
        EarlyStopping(monitor='Valid/mAP', mode='max', patience=args.patience),
        StochasticWeightAveraging(swa_lrs=args.lrE),
        output_mgmt_callback
    ]
    
    return callbacks_list


def get_base_trainer_params(args, callbacks_list, logger_to_use):
    """Get base trainer parameters common to both training and testing."""
    return {
        'deterministic': args.deterministic,
        'devices': args.devices,
        'accelerator': args.accelerator,
        'strategy': args.strategy,
        'precision': args.precision, 
        'sync_batchnorm': args.sync_batchnorm,
        'log_every_n_steps': args.log_every_n_steps,
        'callbacks': callbacks_list,
        'logger': logger_to_use,
        'detect_anomaly': args.detect_anomaly
    }

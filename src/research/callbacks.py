import os
import shutil
import pytorch_lightning as pl
from ..utils.utils import zipResults # Assuming utils.py is in the same directory

class OutputManagementCallback(pl.Callback):
    def __init__(self, base_output_dir="lightning_logs", final_results_dir="SoccerViTAC", logger_type="wandb"):
        super().__init__()
        self.base_output_dir = base_output_dir
        self.final_results_dir = final_results_dir
        self.logger_type = logger_type

        self.run_base_dir = None
        self.run_results_dir = None
        self.run_final_dir = None
        self.version = None

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Called when fit, validate, test, or predict begins."""
        if trainer.logger and hasattr(trainer.logger, 'version') and trainer.logger.version is not None:
            self.version = str(trainer.logger.version)
        else:
            self.version = "unknown_version" # Fallback
            pl_module.print(f"Warning: Could not determine logger version. Using '{self.version}'.")

        # Path for intermediate results, e.g., lightning_logs/VERSION
        self.run_base_dir = os.path.join(self.base_output_dir, self.version)
        # Path for results files within the intermediate dir, e.g., lightning_logs/VERSION/results
        self.run_results_dir = os.path.join(self.run_base_dir, "results")
        
        # Final destination for the entire versioned folder, e.g., SoccerViTAC/VERSION
        self.run_final_dir = os.path.join(self.final_results_dir, self.version)

        if trainer.is_global_zero:
            os.makedirs(self.run_results_dir, exist_ok=True)
            # Ensure the parent of the final destination exists for the move later
            # e.g., SoccerViTAC directory
            os.makedirs(os.path.dirname(self.run_final_dir), exist_ok=True)

    def get_output_path_for_split(self, pl_module: pl.LightningModule, trainer: pl.Trainer):
        """
        Gets the output path for the current split (e.g., test, challenge) 
        and creates it if it doesn't exist.
        This path is where individual game JSONs will be saved.
        Example: lightning_logs/VERSION/results/output_test
        """
        current_split = getattr(pl_module.args, '_split', 'default_split')
        if current_split == 'default_split':
            pl_module.print("Warning: pl_module.args._split not found, using 'default_split'.")

        path = os.path.join(self.run_results_dir, f"output_{current_split}")
        
        if trainer.is_global_zero:
            os.makedirs(path, exist_ok=True)
        # Ensure all processes have the path string, even if only global_zero creates it.
        # Distributed strategies might require barrier if non-global_zero processes access files immediately.
        # For now, assuming path string is sufficient for non_global_zero, and file ops are guarded.
        return path

    def on_predict_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Zips prediction results at the end of the prediction epoch."""
        if trainer.is_global_zero:
            current_split = getattr(pl_module.args, '_split', 'default_predict_split')
            # Ensure the specific output directory for the split exists before zipping
            output_dir_for_split = self.get_output_path_for_split(pl_module, trainer) 
            
            zip_filename = f"result_spotting_{current_split}.zip"
            # Zip file will be created inside output_dir_for_split
            zip_path = os.path.join(output_dir_for_split, zip_filename)
            
            zipResults(zip_path=zip_path,
                       target_dir=output_dir_for_split, # Directory containing results_spotting.json files
                       filename="results_spotting.json")
            pl_module.print(f"OutputManagementCallback: Saved prediction zip to {zip_path}")

    def on_test_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Zips test results at the end of the test epoch."""
        if trainer.is_global_zero:
            current_split = getattr(pl_module.args, '_split', 'test') # Usually 'test'
            output_dir_for_split = self.get_output_path_for_split(pl_module, trainer)

            zip_filename = f"result_spotting_{current_split}.zip"
            zip_path = os.path.join(output_dir_for_split, zip_filename)
            
            zipResults(zip_path=zip_path,
                       target_dir=output_dir_for_split,
                       filename="results_spotting.json")
            pl_module.print(f"OutputManagementCallback: Saved test zip to {zip_path}")
            # Note: Actual evaluation calls (evaluate()) and metric logging
            # are expected to remain in LitModel.test_epoch_end, using output_dir_for_split.

    def on_predict_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Moves the prediction output directory to the final location after all prediction."""
        if self.logger_type == 'wandb' and trainer.is_global_zero:
            if self.run_base_dir and os.path.exists(self.run_base_dir):
                # Ensure parent of the final destination exists (e.g., SoccerViTAC)
                # This should have been created in setup, but double-check.
                final_destination_parent = os.path.dirname(self.run_final_dir)
                os.makedirs(final_destination_parent, exist_ok=True)
                
                if os.path.exists(self.run_final_dir):
                    pl_module.print(f"OutputManagementCallback: Warning: Final destination {self.run_final_dir} already exists. Removing it before move.")
                    shutil.rmtree(self.run_final_dir)
                
                pl_module.print(f"OutputManagementCallback: Moving {self.run_base_dir} to {self.run_final_dir}")
                try:
                    shutil.move(self.run_base_dir, self.run_final_dir)
                except Exception as e:
                    pl_module.print(f"OutputManagementCallback: Error moving directory: {e}")
            else:
                pl_module.print(f"OutputManagementCallback: Source directory {self.run_base_dir} not found for move, or run_base_dir not set.")
    
    # on_test_end: As per the plan, this is not implemented yet,
    # as the final move was specific to wandb predict in the original code.
    # If a similar move is needed for test outputs, it would be added here.

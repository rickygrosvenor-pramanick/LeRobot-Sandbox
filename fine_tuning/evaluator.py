"""
Evaluation utility for fine-tuned Isaac-GR00T models.

This module provides functions to evaluate the performance of a
fine-tuned GR00T model on a validation or test dataset.
"""

import logging
from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from gr00t.model.policy import Gr00tPolicy
from gr00t.data.dataset import LeRobotSingleDataset

from .config import ExperimentConfig


class Evaluator:
    """
    Handles evaluation of a fine-tuned GR00T model.
    """
    
    def __init__(self, model_path: str, config: ExperimentConfig):
        """
        Initialize the evaluator.
        
        Args:
            model_path: Path to the fine-tuned model checkpoint.
            config: Experiment configuration.
        """
        self.model_path = Path(model_path)
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.device = torch.device(config.model.device if torch.cuda.is_available() else "cpu")
        
        self.model = self._load_model()
        self.val_loader = self._setup_dataloader()
        
    def _load_model(self) -> Gr00tPolicy:
        """Load the fine-tuned model."""
        self.logger.info(f"Loading model from {self.model_path}")
        
        # This assumes the model was saved in a way that Gr00tPolicy can load
        # The actual implementation would need modality configs and transforms
        try:
            from gr00t.experiment.data_config import DATA_CONFIG_MAP
            data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
            modality_config = data_config.modality_config()
            transforms = data_config.transform()

            model = Gr00tPolicy(
                model_path=str(self.model_path),
                modality_config=modality_config,
                modality_transform=transforms,
                embodiment_tag=self.config.model.embodiment_tag,
                device=self.device
            )
            model.eval()
            return model
        except (ImportError, KeyError) as e:
            self.logger.error(f"Could not load gr00t model: {e}")
            raise
            
    def _setup_dataloader(self):
        """Setup the validation data loader."""
        self.logger.info("Setting up validation dataloader...")
        
        try:
            from gr00t.experiment.data_config import DATA_CONFIG_MAP
            data_config = DATA_CONFIG_MAP["fourier_gr1_arms_only"]
            modality_config = data_config.modality_config()
            transforms = data_config.transform()

            dataset = LeRobotSingleDataset(
                dataset_path=self.config.data.dataset_path,
                modality_configs=modality_config,
                transforms=transforms,
                embodiment_tag=self.config.model.embodiment_tag,
            )
        except (ImportError, KeyError) as e:
            self.logger.error(f"Could not load gr00t data configs: {e}")
            raise

        # For evaluation, we typically don't need to split
        val_loader = DataLoader(
            dataset,
            batch_size=self.config.data.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers
        )
        return val_loader
        
    def evaluate(self, num_episodes: int = 10, plot: bool = True):
        """
        Run evaluation and compute metrics.
        
        Args:
            num_episodes: Number of episodes to evaluate on.
            plot: Whether to generate plots of predicted vs ground truth actions.
        """
        self.logger.info("Starting evaluation...")
        
        all_predictions = []
        all_ground_truths = []
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_loader):
                if i >= num_episodes:
                    break
                
                # The actual inference call would depend on the model's API
                # This is a placeholder for the core logic
                # predicted_action = self.model.get_action(batch)
                
                # Placeholder for predicted and ground truth actions
                predicted_action = torch.randn_like(batch['action'])
                ground_truth_action = batch['action']
                
                all_predictions.append(predicted_action.cpu().numpy())
                all_ground_truths.append(ground_truth_action.cpu().numpy())
        
        if not all_predictions:
            self.logger.warning("No data was evaluated.")
            return
            
        # Concatenate results
        predictions = np.concatenate(all_predictions, axis=0)
        ground_truths = np.concatenate(all_ground_truths, axis=0)
        
        # Compute metrics
        metrics = self._compute_metrics(predictions, ground_truths)
        
        self.logger.info("Evaluation Metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
            
        # Save metrics
        metrics_path = self.model_path.parent / "evaluation_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        self.logger.info(f"Metrics saved to {metrics_path}")
        
        # Generate plots
        if plot:
            self._plot_results(predictions, ground_truths)
            
    def _compute_metrics(self, predictions: np.ndarray, ground_truths: np.ndarray) -> dict:
        """
        Compute evaluation metrics.
        
        Args:
            predictions: Predicted actions.
            ground_truths: Ground truth actions.
            
        Returns:
            Dictionary of computed metrics.
        """
        mse = np.mean((predictions - ground_truths) ** 2)
        mae = np.mean(np.abs(predictions - ground_truths))
        
        # Per-dimension MSE
        mse_per_dim = np.mean((predictions - ground_truths) ** 2, axis=0)
        
        metrics = {
            'mse': mse,
            'mae': mae,
        }
        
        for i, dim_mse in enumerate(mse_per_dim):
            metrics[f'mse_dim_{i}'] = dim_mse
            
        return metrics
        
    def _plot_results(self, predictions: np.ndarray, ground_truths: np.ndarray):
        """
        Plot predicted vs ground truth actions.
        
        Args:
            predictions: Predicted actions.
            ground_truths: Ground truth actions.
        """
        self.logger.info("Generating plots...")
        
        num_dims = predictions.shape[1]
        
        fig, axes = plt.subplots(num_dims, 1, figsize=(12, 4 * num_dims), sharex=True)
        if num_dims == 1:
            axes = [axes]
            
        for i in range(num_dims):
            axes[i].plot(ground_truths[:, i], label='Ground Truth', color='blue', alpha=0.7)
            axes[i].plot(predictions[:, i], label='Prediction', color='red', linestyle='--')
            axes[i].set_ylabel(f'Action Dim {i}')
            axes[i].legend()
            axes[i].grid(True)
            
        axes[-1].set_xlabel('Timestep')
        fig.suptitle('Predicted vs. Ground Truth Actions', fontsize=16)
        
        plot_path = self.model_path.parent / "evaluation_plot.png"
        plt.savefig(plot_path)
        self.logger.info(f"Plot saved to {plot_path}")
        plt.close()

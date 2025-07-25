"""
Configuration management for Isaac-GR00T fine-tuning.

This module provides configuration classes and utilities for managing
training, model, and data configurations for fine-tuning Isaac-GR00T.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


@dataclass
class ModelConfig:
    """Configuration for the GR00T model."""
    model_path: str = "nvidia/GR00T-N1.5-3B"
    embodiment_tag: str = "gr1"
    device: str = "cuda"
    dtype: str = "float16"
    
    # LoRA configuration
    use_lora: bool = False
    lora_rank: int = 64
    lora_alpha: int = 128
    
    # Model-specific settings
    tune_diffusion_model: bool = True
    tune_vision_encoder: bool = False
    tune_language_model: bool = False


@dataclass  
class DataConfig:
    """Configuration for dataset loading and processing."""
    dataset_path: Union[str, Path] = ""
    validation_split: float = 0.1
    
    # Sequence and action configuration
    sequence_length: int = 10
    action_horizon: int = 4
    delta_indices: List[int] = field(default_factory=lambda: [-1, 0, 1, 2, 3])
    
    # Data loading
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    image_augmentation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    # Basic training settings
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    use_scheduler: bool = True
    scheduler_type: str = "cosine"  # cosine, linear, constant
    warmup_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.0
    
    # Checkpointing and logging
    save_every_n_epochs: int = 5
    eval_every_n_epochs: int = 1
    log_every_n_steps: int = 100
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "gr00t_finetune"
    output_dir: Union[str, Path] = "./outputs"
    seed: int = 42
    
    # Hardware settings
    num_gpus: int = 1
    mixed_precision: bool = True
    compile_model: bool = False


class ConfigManager:
    """Manager for loading and saving experiment configurations."""
    
    @staticmethod
    def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return ConfigManager._dict_to_config(config_dict)
    
    @staticmethod
    def save_config(config: ExperimentConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        config_dict = ConfigManager._config_to_dict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    @staticmethod
    def _dict_to_config(config_dict: Dict[str, Any]) -> ExperimentConfig:
        """Convert dictionary to ExperimentConfig."""
        model_config = ModelConfig(**config_dict.get('model', {}))
        data_config = DataConfig(**config_dict.get('data', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        
        experiment_dict = {k: v for k, v in config_dict.items() 
                          if k not in ['model', 'data', 'training']}
        
        return ExperimentConfig(
            model=model_config,
            data=data_config,
            training=training_config,
            **experiment_dict
        )
    
    @staticmethod
    def _config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
        """Convert ExperimentConfig to dictionary."""
        return {
            'model': ConfigManager._dataclass_to_dict(config.model),
            'data': ConfigManager._dataclass_to_dict(config.data),
            'training': ConfigManager._dataclass_to_dict(config.training),
            'experiment_name': config.experiment_name,
            'output_dir': str(config.output_dir),
            'seed': config.seed,
            'num_gpus': config.num_gpus,
            'mixed_precision': config.mixed_precision,
            'compile_model': config.compile_model
        }
    
    @staticmethod
    def _dataclass_to_dict(obj) -> Dict[str, Any]:
        """Convert dataclass to dictionary."""
        result = {}
        for field_name, field_value in obj.__dict__.items():
            if isinstance(field_value, Path):
                result[field_name] = str(field_value)
            else:
                result[field_name] = field_value
        return result


def create_default_configs() -> Dict[str, ExperimentConfig]:
    """Create default configurations for different scenarios."""
    
    # Base configuration
    base_config = ExperimentConfig()
    
    # High-performance configuration (H100/A100)
    hp_config = ExperimentConfig(
        model=ModelConfig(
            tune_diffusion_model=True,
            tune_vision_encoder=True,
        ),
        data=DataConfig(
            batch_size=16,
            num_workers=8,
        ),
        training=TrainingConfig(
            num_epochs=100,
            learning_rate=2e-4,
            gradient_accumulation_steps=2,
        ),
        num_gpus=1,
        mixed_precision=True,
    )
    
    # Consumer GPU configuration (RTX 4090)
    consumer_config = ExperimentConfig(
        model=ModelConfig(
            tune_diffusion_model=False,  # Disable to save memory
            use_lora=True,
            lora_rank=32,
        ),
        data=DataConfig(
            batch_size=4,
            num_workers=4,
        ),
        training=TrainingConfig(
            num_epochs=50,
            learning_rate=1e-4,
            gradient_accumulation_steps=8,
        ),
        num_gpus=1,
        mixed_precision=True,
    )
    
    # Multi-GPU configuration
    multi_gpu_config = ExperimentConfig(
        model=ModelConfig(
            tune_diffusion_model=True,
        ),
        data=DataConfig(
            batch_size=32,  # Total batch size across GPUs
            num_workers=16,
        ),
        training=TrainingConfig(
            num_epochs=75,
            learning_rate=3e-4,
            gradient_accumulation_steps=1,
        ),
        num_gpus=4,
        mixed_precision=True,
    )
    
    # Development/debugging configuration
    debug_config = ExperimentConfig(
        model=ModelConfig(
            tune_diffusion_model=False,
            use_lora=True,
            lora_rank=16,
        ),
        data=DataConfig(
            batch_size=2,
            num_workers=1,
        ),
        training=TrainingConfig(
            num_epochs=5,
            learning_rate=1e-4,
            gradient_accumulation_steps=2,
            save_every_n_epochs=1,
            eval_every_n_epochs=1,
            log_every_n_steps=10,
        ),
        num_gpus=1,
        mixed_precision=False,
    )
    
    return {
        'base': base_config,
        'high_performance': hp_config,
        'consumer_gpu': consumer_config,
        'multi_gpu': multi_gpu_config,
        'debug': debug_config,
    }


def get_embodiment_configs() -> Dict[str, Dict[str, Any]]:
    """Get predefined embodiment configurations."""
    return {
        'gr1': {
            'state_keys': {
                'left_arm': 7,
                'left_hand': 6, 
                'right_arm': 7,
                'right_hand': 6,
                'torso': 3,
                'head': 2,
                'legs': 12,
            },
            'action_keys': {
                'left_arm': 7,
                'left_hand': 6,
                'right_arm': 7, 
                'right_hand': 6,
                'torso': 3,
                'head': 2,
                'legs': 12,
            },
            'camera_names': ['head_camera', 'chest_camera'],
            'embodiment_tag': 'gr1'
        },
        'oxe_droid': {
            'state_keys': {
                'arm_joints': 7,
                'gripper': 2,
                'ee_position': 3,
                'ee_orientation': 4,
            },
            'action_keys': {
                'ee_delta_position': 3,
                'ee_delta_orientation': 3,
                'gripper': 1,
            },
            'camera_names': ['wrist_camera', 'shoulder_camera'],
            'embodiment_tag': 'oxe_droid'
        },
        'agibot_genie1': {
            'state_keys': {
                'left_arm': 7,
                'right_arm': 7,
                'left_gripper': 1,
                'right_gripper': 1,
                'torso': 3,
                'head': 2,
            },
            'action_keys': {
                'left_arm': 7,
                'right_arm': 7,
                'left_gripper': 1,
                'right_gripper': 1,
                'torso': 3,
                'head': 2,
            },
            'camera_names': ['head_camera', 'chest_camera'],
            'embodiment_tag': 'agibot_genie1'
        }
    }

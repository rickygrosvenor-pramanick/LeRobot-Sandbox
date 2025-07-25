"""
Example script for fine-tuning Isaac-GR00T on custom datasets.

This script demonstrates how to fine-tune the GR00T N1.5 model
on your converted LeRobot-format datasets.
"""

import logging
import os
from pathlib import Path
from typing import Optional

from fine_tuning.config import (
    ConfigManager, 
    ExperimentConfig, 
    create_default_configs,
    get_embodiment_configs
)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_environment():
    """Setup environment variables and dependencies."""
    print("Setting up environment...")
    
    # Check for CUDA availability
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA available with {torch.cuda.device_count()} GPU(s)")
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - will use CPU (not recommended)")
    except ImportError:
        print("❌ PyTorch not installed")
        return False
    
    # Set environment variables for optimal performance
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    
    return True


def create_experiment_config(
    dataset_path: str,
    embodiment_tag: str = "gr1",
    config_type: str = "base",
    experiment_name: Optional[str] = None
) -> ExperimentConfig:
    """Create experiment configuration for fine-tuning."""
    
    # Get default config based on type
    default_configs = create_default_configs()
    config = default_configs.get(config_type, default_configs['base'])
    
    # Update with dataset-specific settings
    config.data.dataset_path = dataset_path
    config.model.embodiment_tag = embodiment_tag
    
    if experiment_name:
        config.experiment_name = experiment_name
    else:
        config.experiment_name = f"gr00t_finetune_{embodiment_tag}_{config_type}"
    
    # Set output directory
    config.output_dir = Path("./outputs") / config.experiment_name
    
    return config


def finetune_basic_example():
    """Basic fine-tuning example with default settings."""
    print("\n🚀 Basic Fine-tuning Example")
    print("=" * 40)
    
    # Configuration
    dataset_path = "./output_data/lerobot_dataset_rosbag"
    embodiment_tag = "gr1"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        print("   Please run convert_data_example.py first to create a dataset")
        return
    
    # Create configuration
    config = create_experiment_config(
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        config_type="base",
        experiment_name="basic_finetune"
    )
    
    # Save configuration
    config_path = Path("./configs/experiments/basic_finetune.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ConfigManager.save_config(config, config_path)
    
    print(f"📝 Configuration saved to: {config_path}")
    print(f"🎯 Experiment: {config.experiment_name}")
    print(f"📊 Dataset: {config.data.dataset_path}")
    print(f"🤖 Embodiment: {config.model.embodiment_tag}")
    print(f"🔧 Model: {config.model.model_path}")
    
    # Simulate fine-tuning (replace with actual training code)
    print("\n📈 Starting fine-tuning...")
    print("   Note: This is a simulation. In practice, you would:")
    print("   1. Load the dataset using LeRobotSingleDataset")
    print("   2. Initialize the GR00T model and policy")
    print("   3. Run the training loop with the specified configuration")
    print("   4. Save checkpoints and monitor training progress")
    
    print(f"\n✅ Configuration ready for fine-tuning!")
    print(f"   Run: python scripts/gr00t_finetune.py --config {config_path}")


def finetune_consumer_gpu_example():
    """Fine-tuning example optimized for consumer GPUs (RTX 4090)."""
    print("\n🖥️  Consumer GPU Fine-tuning Example")
    print("=" * 45)
    
    dataset_path = "./output_data/lerobot_dataset_hdf5"
    embodiment_tag = "oxe_droid"
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found at: {dataset_path}")
        return
    
    # Create consumer GPU optimized configuration
    config = create_experiment_config(
        dataset_path=dataset_path,
        embodiment_tag=embodiment_tag,
        config_type="consumer_gpu",
        experiment_name="consumer_gpu_finetune"
    )
    
    # Additional optimizations for consumer GPUs
    config.model.tune_diffusion_model = False  # Save memory
    config.model.use_lora = True
    config.model.lora_rank = 32
    config.data.batch_size = 2  # Smaller batch size
    config.training.gradient_accumulation_steps = 16  # Compensate with gradient accumulation
    
    config_path = Path("./configs/experiments/consumer_gpu_finetune.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ConfigManager.save_config(config, config_path)
    
    print(f"📝 Configuration saved to: {config_path}")
    print("🔧 Optimizations for consumer GPU:")
    print(f"   - LoRA fine-tuning: {config.model.use_lora}")
    print(f"   - LoRA rank: {config.model.lora_rank}")
    print(f"   - Batch size: {config.data.batch_size}")
    print(f"   - Gradient accumulation: {config.training.gradient_accumulation_steps}")
    print(f"   - Diffusion tuning disabled: {not config.model.tune_diffusion_model}")
    
    print(f"\n✅ Consumer GPU configuration ready!")
    print("   Add --no-tune_diffusion_model flag when running on RTX 4090")


def finetune_custom_embodiment_example():
    """Fine-tuning example for a custom robot embodiment."""
    print("\n🤖 Custom Embodiment Fine-tuning Example")
    print("=" * 45)
    
    # Define custom embodiment configuration
    custom_embodiment_config = {
        'state_keys': {
            'left_arm': 7,
            'right_arm': 7,
            'torso': 3,
            'head': 2,
            'mobile_base': 3,
            'left_gripper': 2,
            'right_gripper': 2
        },
        'action_keys': {
            'left_arm': 7,
            'right_arm': 7, 
            'torso': 3,
            'head': 2,
            'mobile_base': 3,
            'left_gripper': 1,
            'right_gripper': 1
        },
        'camera_names': ['head_camera', 'chest_camera', 'wrist_camera'],
        'embodiment_tag': 'new_embodiment'
    }
    
    dataset_path = "./output_data/custom_robot_dataset"
    
    # Create configuration for new embodiment
    config = create_experiment_config(
        dataset_path=dataset_path,
        embodiment_tag="new_embodiment",
        config_type="base",
        experiment_name="custom_embodiment_finetune"
    )
    
    # Custom embodiment requires training new action head
    config.model.embodiment_tag = "new_embodiment"
    config.training.num_epochs = 100  # May need more training
    config.training.learning_rate = 5e-5  # Lower learning rate for stability
    
    config_path = Path("./configs/experiments/custom_embodiment_finetune.yaml")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    ConfigManager.save_config(config, config_path)
    
    # Save embodiment configuration
    embodiment_config_path = Path("./configs/embodiments/custom_robot.yaml")
    embodiment_config_path.parent.mkdir(parents=True, exist_ok=True)
    
    import yaml
    with open(embodiment_config_path, 'w') as f:
        yaml.dump(custom_embodiment_config, f, default_flow_style=False, indent=2)
    
    print(f"📝 Configuration saved to: {config_path}")
    print(f"🤖 Embodiment config saved to: {embodiment_config_path}")
    print("🔧 Custom embodiment settings:")
    print(f"   - State dimensions: {sum(custom_embodiment_config['state_keys'].values())}")
    print(f"   - Action dimensions: {sum(custom_embodiment_config['action_keys'].values())}")
    print(f"   - Camera views: {len(custom_embodiment_config['camera_names'])}")
    print(f"   - Training epochs: {config.training.num_epochs}")
    
    print(f"\n✅ Custom embodiment configuration ready!")
    print("   Note: New embodiments require training from scratch for the action head")


def evaluate_model_example():
    """Example of evaluating a fine-tuned model."""
    print("\n📊 Model Evaluation Example")
    print("=" * 35)
    
    model_path = "./outputs/basic_finetune/checkpoints/final_model"
    dataset_path = "./output_data/lerobot_dataset_rosbag"
    
    print("🔍 Evaluation steps:")
    print("1. Load the fine-tuned model")
    print("2. Load validation dataset")
    print("3. Run inference on validation episodes")
    print("4. Compute metrics (MSE, success rate, etc.)")
    print("5. Generate evaluation report")
    
    # Example evaluation configuration
    eval_config = {
        'model_path': model_path,
        'dataset_path': dataset_path,
        'metrics': ['mse', 'mae', 'success_rate'],
        'num_episodes': 10,
        'save_videos': True,
        'output_dir': './outputs/evaluation'
    }
    
    print(f"\n📈 Example evaluation command:")
    print(f"python scripts/eval_policy.py \\")
    print(f"  --model-path {model_path} \\")
    print(f"  --dataset-path {dataset_path} \\")
    print(f"  --plot --num-episodes 10")


def deployment_example():
    """Example of deploying a fine-tuned model."""
    print("\n🚀 Model Deployment Example")
    print("=" * 35)
    
    print("🔧 Deployment options:")
    print("1. PyTorch inference (development/testing)")
    print("2. TensorRT optimization (production)")
    print("3. Jetson deployment (edge devices)")
    print("4. Real-time robot control")
    
    print("\n📝 PyTorch deployment:")
    print("```python")
    print("from gr00t.model.policy import Gr00tPolicy")
    print("")
    print("policy = Gr00tPolicy(")
    print("    model_path='./outputs/basic_finetune/final_model',")
    print("    embodiment_tag='gr1'")
    print(")")
    print("")
    print("action = policy.get_action(observation)")
    print("```")
    
    print("\n⚡ TensorRT optimization:")
    print("python deployment_scripts/convert_to_tensorrt.py \\")
    print("  --model-path ./outputs/basic_finetune/final_model \\")
    print("  --output-path ./outputs/tensorrt_model")


def main():
    """Main function to run fine-tuning examples."""
    setup_logging()
    
    print("🎯 Isaac-GR00T Fine-tuning Examples")
    print("=" * 50)
    
    # Setup environment
    if not setup_environment():
        print("❌ Environment setup failed")
        return
    
    # Create output directories
    output_dirs = [
        Path("./configs/experiments"),
        Path("./configs/embodiments"),
        Path("./outputs")
    ]
    
    for output_dir in output_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run examples
    try:
        # Example 1: Basic fine-tuning
        finetune_basic_example()
        
        # Example 2: Consumer GPU optimization
        finetune_consumer_gpu_example()
        
        # Example 3: Custom embodiment
        finetune_custom_embodiment_example()
        
        # Example 4: Model evaluation
        evaluate_model_example()
        
        # Example 5: Deployment
        deployment_example()
        
    except Exception as e:
        print(f"❌ Error during fine-tuning setup: {e}")
        logging.exception("Fine-tuning setup error")
    
    print("\n📖 Next Steps:")
    print("1. Install Isaac-GR00T following the installation guide")
    print("2. Convert your data using convert_data_example.py")
    print("3. Choose appropriate configuration based on your hardware")
    print("4. Run fine-tuning with the generated configuration files")
    print("5. Evaluate and deploy your fine-tuned model")
    print("\n📚 For more details, see the README.md and documentation")


if __name__ == "__main__":
    main()

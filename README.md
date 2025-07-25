# Isaac-GR00T Fine-tuning with LeRobot Data Conversion

This repository provides a modular framework for fine-tuning NVIDIA's Isaac-GR00T N1.5 model with your own robot data. It includes comprehensive data conversion utilities to transform various data formats into the required LeRobot-compatible schema.

## 🎯 Overview

Isaac-GR00T N1.5 is NVIDIA's foundation model for generalized humanoid robot reasoning and skills. This framework enables you to:

1. **Convert your robot data** to LeRobot-compatible format
2. **Fine-tune** the pre-trained GR00T N1.5 model
3. **Deploy** the model for inference on your robot

## 📋 Prerequisites

### Hardware Requirements
- **For Fine-tuning**: H100, L40, RTX 4090, or A6000 GPU
- **For Inference**: RTX 3090, RTX 4090, or A6000 GPU
- **Memory**: Sufficient RAM for your dataset size

### Software Requirements
- Ubuntu 20.04 or 22.04
- Python 3.10
- CUDA 12.4
- TensorRT (optional, for deployment)

### System Dependencies
```bash
sudo apt-get update
sudo apt-get install ffmpeg libsm6 libxext6
```

## 🗂️ Data Requirements

To fine-tune Isaac-GR00T, you need robot demonstration data in the form of `(video, state, action)` triplets:

### Required Data Components

1. **Video Observations**
   - Camera feeds (RGB images)
   - Format: MP4 files
   - Multiple camera angles supported

2. **State Information**
   - Joint positions, velocities
   - End-effector poses
   - Sensor readings
   - Any proprioceptive data

3. **Actions**
   - Joint commands
   - End-effector commands
   - Gripper commands

4. **Language Annotations** (optional but recommended)
   - Task descriptions
   - Natural language instructions
   - Validity labels

### Supported Robot Embodiments

GR00T N1.5 provides three pre-trained embodiment heads:

- **`EmbodimentTag.GR1`**: Humanoid robots with dexterous hands (absolute joint space)
- **`EmbodimentTag.OXE_DROID`**: Single arm robots (delta end-effector control)
- **`EmbodimentTag.AGIBOT_GENIE1`**: Humanoid robots with grippers (absolute joint space)
- **`EmbodimentTag.NEW_EMBODIMENT`**: Custom embodiments (requires training)

## 🔧 Installation

1. **Clone the Isaac-GR00T repository**:
```bash
git clone https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T
```

2. **Create conda environment**:
```bash
conda create -n gr00t python=3.10
conda activate gr00t
```

3. **Install dependencies**:
```bash
pip install --upgrade setuptools
pip install -e .[base]
pip install --no-build-isolation flash-attn==2.7.1.post4
```

## 📁 Project Structure

```
LeRobot-Sandbox/
├── data_conversion/          # Data conversion utilities
│   ├── __init__.py
│   ├── base_converter.py     # Base converter class
│   ├── converters/           # Format-specific converters
│   │   ├── __init__.py
│   │   ├── rosbag_converter.py
│   │   ├── hdf5_converter.py
│   │   ├── pickle_converter.py
│   │   └── custom_converter.py
│   └── validators/           # Data validation utilities
│       ├── __init__.py
│       └── schema_validator.py
├── fine_tuning/              # Fine-tuning utilities
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   ├── trainer.py           # Training logic
│   └── evaluator.py         # Evaluation utilities
├── examples/                 # Example scripts and data
│   ├── convert_data_example.py
│   ├── finetune_example.py
│   └── sample_data/         # Sample datasets
├── configs/                  # Configuration files
│   ├── embodiments/         # Embodiment-specific configs
│   └── training/            # Training configurations
├── scripts/                  # Utility scripts
│   ├── setup_environment.py
│   ├── validate_dataset.py
│   └── benchmark_model.py
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### 1. Data Conversion

Convert your robot data to LeRobot format:

```python
from data_conversion.converters import RosbagConverter

# Initialize converter
converter = RosbagConverter()

# Convert your data
converter.convert(
    input_path="/path/to/your/rosbags",
    output_path="/path/to/lerobot/dataset",
    config={
        "camera_topics": ["/camera/rgb/image_raw"],
        "state_topics": ["/joint_states"],
        "action_topics": ["/cmd_joint_positions"],
        "language_annotations": "/task/description"
    }
)
```

### 2. Fine-tuning

Fine-tune the model on your converted data:

```python
from fine_tuning.trainer import GR00TTrainer

trainer = GR00TTrainer(
    model_path="nvidia/GR00T-N1.5-3B",
    dataset_path="/path/to/lerobot/dataset",
    embodiment_tag="gr1",  # or your custom embodiment
    config_path="configs/training/default.yaml"
)

trainer.train()
```

### 3. Inference

Use your fine-tuned model:

```python
from gr00t.model.policy import Gr00tPolicy

policy = Gr00tPolicy(
    model_path="/path/to/fine-tuned/model",
    embodiment_tag="gr1"
)

# Get action from observation
action = policy.get_action(observation)
```

## 📊 Data Format Details

The LeRobot-compatible format requires specific structure and metadata. See the [Data Conversion Guide](docs/data_conversion_guide.md) for detailed specifications.

### Key Files Required:
- `meta/modality.json` - Defines data modalities and structure
- `meta/tasks.jsonl` - Task descriptions
- `meta/episodes.jsonl` - Episode metadata
- `meta/info.json` - Dataset information
- `data/chunk-*/episode_*.parquet` - Trajectory data
- `videos/chunk-*/observation.images.*/episode_*.mp4` - Video data

## 🎛️ Configuration

The framework supports flexible configuration through YAML files:

```yaml
# configs/training/default.yaml
model:
  name: "nvidia/GR00T-N1.5-3B"
  embodiment_tag: "gr1"

training:
  batch_size: 8
  learning_rate: 1e-4
  num_epochs: 50
  gradient_accumulation_steps: 4

data:
  sequence_length: 10
  action_horizon: 4
  delta_indices: [-1, 0, 1, 2, 3]
```

## 🔍 Validation and Debugging

Validate your converted dataset:

```bash
python scripts/validate_dataset.py --dataset-path /path/to/lerobot/dataset
```

## 📚 Examples and Tutorials

- [Basic Data Conversion](examples/convert_data_example.py)
- [Fine-tuning Tutorial](examples/finetune_example.py)
- [Custom Embodiment Setup](docs/custom_embodiment_guide.md)
- [Deployment Guide](docs/deployment_guide.md)

## 🐛 Troubleshooting

### Common Issues:

1. **CUDA OOM during fine-tuning**: Use `--no-tune_diffusion_model` flag for RTX 4090
2. **Data format errors**: Run validation script to check schema compliance
3. **Missing metadata**: Ensure all required metadata files are present

## 📖 Additional Resources

- [Isaac-GR00T Official Repository](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1.5 Model on HuggingFace](https://huggingface.co/nvidia/GR00T-N1.5-3B)
- [Sample Dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim)
- [Technical Paper](https://arxiv.org/abs/2503.14734)

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

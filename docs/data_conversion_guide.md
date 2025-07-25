# Data Conversion Guide for Isaac-GR00T

This guide provides detailed instructions for converting your robot data to the LeRobot-compatible format required for Isaac-GR00T fine-tuning.

## Overview

Isaac-GR00T requires robot demonstration data in a specific LeRobot-compatible format that includes:
- **Video observations** from robot cameras
- **State information** (joint positions, end-effector poses, etc.)
- **Action commands** (joint commands, end-effector commands)
- **Language annotations** (task descriptions, instructions)

## Supported Input Formats

The framework supports conversion from multiple common robot data formats:

### 1. ROS Bag Files
- **Format**: `.bag` files from ROS1/ROS2
- **Contains**: Timestamped messages from various topics
- **Converter**: `RosbagConverter`

### 2. HDF5 Files  
- **Format**: `.hdf5`, `.h5` files
- **Contains**: Hierarchical data with observations, actions, videos
- **Converter**: `HDF5Converter`
- **Compatible with**: Robomimic, D4RL, RLDS datasets

### 3. Custom Formats
- **Format**: Any format (pickle, JSON, etc.)
- **Converter**: Extend `BaseConverter` class

## Output Format: LeRobot-Compatible Schema

The converted data follows the LeRobot V2.0 format with additional GR00T-specific metadata:

```
dataset/
├── meta/
│   ├── modality.json     # Data modality definitions
│   ├── tasks.jsonl       # Task descriptions
│   ├── episodes.jsonl    # Episode metadata
│   └── info.json         # Dataset information
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── episode_000001.parquet
└── videos/
    └── chunk-000/
        ├── observation.images.camera1/
        │   ├── episode_000000.mp4
        │   └── episode_000001.mp4
        └── observation.images.camera2/
            ├── episode_000000.mp4
            └── episode_000001.mp4
```

## Key Metadata Files

### modality.json
Defines how state and action data is structured:

```json
{
  "state": {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "gripper": {"start": 14, "end": 16}
  },
  "action": {
    "left_arm": {"start": 0, "end": 7, "absolute": true},
    "right_arm": {"start": 7, "end": 14, "absolute": true}
  },
  "video": {
    "head_camera": {"original_key": "observation.images.head_camera"}
  },
  "annotation": {
    "human.action.task_description": {},
    "human.validity": {}
  }
}
```

### Parquet Data Format
Each episode is stored as a parquet file with columns:
- `observation.state`: Concatenated state vector
- `action`: Concatenated action vector  
- `timestamp`: Time in seconds
- `episode_index`: Episode identifier
- `annotation.*`: Language annotations

## Conversion Examples

### ROS Bag Conversion

```python
from data_conversion.converters.rosbag_converter import RosbagConverter

# Configure robot embodiment
embodiment_config = {
    'state_keys': {
        'left_arm': 7,
        'right_arm': 7,
        'gripper': 2
    },
    'action_keys': {
        'left_arm': 7,
        'right_arm': 7,
        'gripper': 2
    },
    'camera_names': ['head_camera', 'wrist_camera'],
    'embodiment_tag': 'gr1'
}

# Configure ROS topics
topic_config = {
    'camera_topics': ['/head_camera/image_raw', '/wrist_camera/image_raw'],
    'state_topics': ['/joint_states'],
    'action_topics': ['/joint_commands'],
    'language_topics': ['/task_description']
}

# Convert data
converter = RosbagConverter(embodiment_config, topic_config)
converter.convert(
    input_path="./rosbags",
    output_path="./lerobot_dataset",
    language_annotations={"demo1.bag": "Pick up the red block"}
)
```

### HDF5 Conversion

```python
from data_conversion.converters.hdf5_converter import HDF5Converter

# Configure for robomimic-style data
embodiment_config = {
    'state_keys': {'robot0_eef_pos': 3, 'robot0_joint_pos': 7},
    'action_keys': {'robot0_eef_pos': 3, 'robot0_gripper': 1},
    'camera_names': ['agentview_image'],
    'embodiment_tag': 'oxe_droid'
}

hdf5_structure = {
    'observations_state': '/data/obs/robot0_eef_pos',
    'actions': '/data/actions',
    'images_base': '/data/obs'
}

converter = HDF5Converter(embodiment_config, hdf5_structure)
converter.convert("./hdf5_data", "./lerobot_dataset")
```

## Embodiment Configurations

### Humanoid Robot (GR1)
```python
{
    'state_keys': {
        'left_arm': 7, 'left_hand': 6,
        'right_arm': 7, 'right_hand': 6,
        'torso': 3, 'head': 2, 'legs': 12
    },
    'action_keys': {
        'left_arm': 7, 'left_hand': 6,
        'right_arm': 7, 'right_hand': 6,
        'torso': 3, 'head': 2, 'legs': 12
    },
    'embodiment_tag': 'gr1'
}
```

### Single-Arm Robot (OXE-DROID)
```python
{
    'state_keys': {
        'arm_joints': 7, 'gripper': 2,
        'ee_position': 3, 'ee_orientation': 4
    },
    'action_keys': {
        'ee_delta_position': 3,
        'ee_delta_orientation': 3,
        'gripper': 1
    },
    'embodiment_tag': 'oxe_droid'
}
```

## Validation

After conversion, validate your dataset:

```bash
python scripts/validate_dataset.py --dataset-path ./lerobot_dataset
```

This checks:
- ✅ Directory structure
- ✅ Required metadata files
- ✅ Data format consistency
- ✅ Video file integrity
- ✅ Modality configuration

## Troubleshooting

### Common Issues

1. **Missing Dependencies**
   ```bash
   pip install h5py rosbag cv_bridge  # For ROS/HDF5 support
   ```

2. **Video Conversion Errors**
   - Ensure OpenCV is installed: `pip install opencv-python`
   - Check video codec support

3. **State/Action Dimension Mismatch**
   - Verify embodiment configuration matches your robot
   - Check start/end indices in modality.json

4. **Memory Issues with Large Datasets**
   - Reduce chunk_size parameter
   - Process datasets in smaller batches

### Data Quality Tips

1. **Synchronization**: Ensure camera, state, and action data are properly synchronized
2. **Frequency**: Match control frequency with video frame rate when possible
3. **Coverage**: Include diverse scenarios and robot configurations
4. **Language**: Provide meaningful task descriptions for better language grounding

## Next Steps

After successful conversion:

1. **Validate** the dataset using the validation script
2. **Configure** your embodiment settings
3. **Start fine-tuning** with the converted dataset
4. **Evaluate** the trained model performance

For more examples, see `examples/convert_data_example.py`.

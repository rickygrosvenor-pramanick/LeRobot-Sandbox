"""
Example script for converting robot data to LeRobot format.

This script demonstrates how to use the data conversion utilities
to transform various robot data formats for Isaac-GR00T fine-tuning.
"""

import logging
from pathlib import Path
from typing import Dict, Any

# Import converters
from data_conversion.converters.rosbag_converter import RosbagConverter, create_humanoid_config, create_topic_config
from data_conversion.converters.hdf5_converter import HDF5Converter, create_robomimic_structure


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def convert_rosbag_data():
    """Example of converting ROS bag data."""
    print("Converting ROS bag data...")
    
    # Configuration for humanoid robot
    embodiment_config = create_humanoid_config()
    topic_config = create_topic_config()
    
    # Update topic config for your specific robot
    topic_config.update({
        'camera_topics': [
            '/head_camera/image_raw',
            '/chest_camera/image_raw',
            '/wrist_camera/image_raw'
        ],
        'state_topics': [
            '/joint_states',
            '/robot_state'
        ],
        'action_topics': [
            '/joint_commands',
            '/cartesian_commands'
        ],
        'language_topics': [
            '/task_description'
        ]
    })
    
    # Language annotations (optional)
    language_annotations = {
        "demo_001.bag": "Pick up the red block and place it in the blue box",
        "demo_002.bag": "Pour water from the bottle into the cup",
        "demo_003.bag": "Open the drawer and retrieve the object inside"
    }
    
    # Initialize converter
    converter = RosbagConverter(
        embodiment_config=embodiment_config,
        topic_config=topic_config,
        video_fps=30,
        chunk_size=100
    )
    
    # Convert data
    input_path = Path("./input_data/rosbags")
    output_path = Path("./output_data/lerobot_dataset_rosbag")
    
    try:
        converter.convert(
            input_path=input_path,
            output_path=output_path,
            language_annotations=language_annotations,
            validation_split=0.1
        )
        print(f"✅ ROS bag conversion completed! Output saved to: {output_path}")
    except Exception as e:
        print(f"❌ ROS bag conversion failed: {e}")


def convert_hdf5_data():
    """Example of converting HDF5 data."""
    print("Converting HDF5 data...")
    
    # Configuration for manipulation robot
    embodiment_config = {
        'state_keys': {
            'robot0_eef_pos': 3,
            'robot0_eef_quat': 4,
            'robot0_gripper_qpos': 2,
            'robot0_joint_pos': 7
        },
        'action_keys': {
            'robot0_eef_pos': 3,
            'robot0_eef_quat': 4,
            'robot0_gripper_qpos': 2
        },
        'camera_names': ['agentview_image', 'robot0_eye_in_hand_image'],
        'embodiment_tag': 'oxe_droid'
    }
    
    # HDF5 structure for robomimic-style data
    hdf5_structure = create_robomimic_structure()
    hdf5_structure.update({
        'observations_state': '/data/obs/robot0_eef_pos',
        'actions': '/data/actions',
        'images_base': '/data/obs'
    })
    
    # Initialize converter
    converter = HDF5Converter(
        embodiment_config=embodiment_config,
        hdf5_structure=hdf5_structure,
        video_fps=20,
        chunk_size=50
    )
    
    # Convert data
    input_path = Path("./input_data/hdf5_files")
    output_path = Path("./output_data/lerobot_dataset_hdf5")
    
    try:
        converter.convert(
            input_path=input_path,
            output_path=output_path,
            validation_split=0.15
        )
        print(f"✅ HDF5 conversion completed! Output saved to: {output_path}")
    except Exception as e:
        print(f"❌ HDF5 conversion failed: {e}")


def convert_custom_data():
    """Example of converting custom data format."""
    print("Converting custom data format...")
    
    # For custom formats, you would create a custom converter
    # by subclassing BaseConverter and implementing the required methods
    
    # Example configuration for a custom robot
    embodiment_config = {
        'state_keys': {
            'joint_positions': 12,
            'joint_velocities': 12,
            'base_pose': 6,
            'gripper_state': 2
        },
        'action_keys': {
            'joint_commands': 12,
            'gripper_command': 1
        },
        'camera_names': ['front_camera', 'side_camera'],
        'embodiment_tag': 'new_embodiment'
    }
    
    print("For custom data formats:")
    print("1. Create a custom converter by subclassing BaseConverter")
    print("2. Implement _load_episode_data() and _extract_states_and_actions()")
    print("3. Handle your specific data format in these methods")
    print("4. See base_converter.py for the interface definition")


def validate_converted_data(dataset_path: Path):
    """Validate that the converted data is properly formatted."""
    print(f"Validating dataset at: {dataset_path}")
    
    required_files = [
        "meta/modality.json",
        "meta/tasks.jsonl", 
        "meta/episodes.jsonl",
        "meta/info.json"
    ]
    
    for required_file in required_files:
        file_path = dataset_path / required_file
        if file_path.exists():
            print(f"✅ Found: {required_file}")
        else:
            print(f"❌ Missing: {required_file}")
    
    # Check for data files
    data_dir = dataset_path / "data"
    if data_dir.exists():
        parquet_files = list(data_dir.glob("**/*.parquet"))
        print(f"✅ Found {len(parquet_files)} parquet files")
    else:
        print("❌ Missing data directory")
    
    # Check for video files
    videos_dir = dataset_path / "videos"
    if videos_dir.exists():
        video_files = list(videos_dir.glob("**/*.mp4"))
        print(f"✅ Found {len(video_files)} video files")
    else:
        print("❌ Missing videos directory")


def main():
    """Main function to run data conversion examples."""
    setup_logging()
    
    print("🤖 Isaac-GR00T Data Conversion Examples")
    print("=" * 50)
    
    # Ensure input directories exist (create dummy structure for demo)
    input_dirs = [
        Path("./input_data/rosbags"),
        Path("./input_data/hdf5_files")
    ]
    
    for input_dir in input_dirs:
        input_dir.mkdir(parents=True, exist_ok=True)
    
    # Run conversion examples
    try:
        # Example 1: ROS bag conversion
        convert_rosbag_data()
        print()
        
        # Example 2: HDF5 conversion
        convert_hdf5_data()
        print()
        
        # Example 3: Custom format info
        convert_custom_data()
        print()
        
        # Validation examples
        output_datasets = [
            Path("./output_data/lerobot_dataset_rosbag"),
            Path("./output_data/lerobot_dataset_hdf5")
        ]
        
        for dataset_path in output_datasets:
            if dataset_path.exists():
                validate_converted_data(dataset_path)
                print()
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")
        logging.exception("Conversion error")
    
    print("📖 Next Steps:")
    print("1. Validate your converted dataset using scripts/validate_dataset.py")
    print("2. Configure your embodiment settings in configs/embodiments/")
    print("3. Start fine-tuning with examples/finetune_example.py")
    print("4. See the README.md for detailed documentation")


if __name__ == "__main__":
    main()

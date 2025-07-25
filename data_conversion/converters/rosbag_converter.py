"""
ROS bag converter for transforming ROS data to LeRobot format.

This converter processes ROS bag files containing robot demonstration data
and converts them to the LeRobot-compatible schema for Isaac-GR00T fine-tuning.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import PoseStamped, Twist
from std_msgs.msg import String

try:
    import rosbag
    ROSBAG_AVAILABLE = True
except ImportError:
    ROSBAG_AVAILABLE = False
    logging.warning("rosbag not available. ROS bag conversion will not work.")

from ..base_converter import BaseConverter


class RosbagConverter(BaseConverter):
    """
    Converter for ROS bag files to LeRobot format.
    
    This converter processes ROS bag files containing:
    - Camera images (sensor_msgs/Image)
    - Joint states (sensor_msgs/JointState) 
    - Action commands (various message types)
    - Language annotations (std_msgs/String)
    """
    
    def __init__(self,
                 embodiment_config: Dict[str, Any],
                 topic_config: Dict[str, Any],
                 video_fps: int = 30,
                 chunk_size: int = 1000):
        """
        Initialize the ROS bag converter.
        
        Args:
            embodiment_config: Robot embodiment configuration
            topic_config: Mapping of data types to ROS topics
            video_fps: Target frame rate for video data
            chunk_size: Number of episodes per chunk
        """
        if not ROSBAG_AVAILABLE:
            raise ImportError("rosbag package is required for RosbagConverter")
            
        super().__init__(embodiment_config, video_fps, chunk_size)
        self.topic_config = topic_config
        self.cv_bridge = CvBridge()
        
        # Validate topic configuration
        self._validate_topic_config()
    
    def _validate_topic_config(self) -> None:
        """Validate the topic configuration."""
        required_keys = ['camera_topics', 'state_topics', 'action_topics']
        for key in required_keys:
            if key not in self.topic_config:
                raise ValueError(f"Missing required key '{key}' in topic_config")
    
    def _get_episode_paths(self, input_path: Path) -> List[Path]:
        """Get list of ROS bag files from input directory."""
        bag_files = list(input_path.glob("*.bag"))
        if not bag_files:
            # Try recursive search
            bag_files = list(input_path.glob("**/*.bag"))
        return sorted(bag_files)
    
    def _load_episode_data(self, episode_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from a ROS bag file.
        
        Args:
            episode_path: Path to the .bag file
            
        Returns:
            Dictionary containing episode data
        """
        bag_path = Path(episode_path)
        
        if not bag_path.exists():
            raise FileNotFoundError(f"Bag file not found: {bag_path}")
        
        episode_data = {
            'observations': [],
            'actions': [],
            'timestamps': [],
            'videos': {},
            'language': []
        }
        
        # Initialize video storage for each camera
        for camera_topic in self.topic_config['camera_topics']:
            camera_name = self._topic_to_camera_name(camera_topic)
            episode_data['videos'][camera_name] = []
        
        # Read bag file
        with rosbag.Bag(str(bag_path), 'r') as bag:
            # Get bag info for synchronization
            bag_info = bag.get_type_and_topic_info()
            
            # Collect all messages by timestamp
            all_messages = []
            
            for topic, msg, timestamp in bag.read_messages():
                all_messages.append({
                    'topic': topic,
                    'message': msg,
                    'timestamp': timestamp.to_sec()
                })
            
            # Sort by timestamp for proper synchronization
            all_messages.sort(key=lambda x: x['timestamp'])
            
            # Process messages
            current_state = {}
            current_action = None
            current_images = {}
            
            for msg_data in all_messages:
                topic = msg_data['topic']
                msg = msg_data['message']
                timestamp = msg_data['timestamp']
                
                # Process camera images
                if topic in self.topic_config['camera_topics']:
                    camera_name = self._topic_to_camera_name(topic)
                    image = self._process_image_message(msg)
                    current_images[camera_name] = image
                
                # Process state information
                elif topic in self.topic_config['state_topics']:
                    state_data = self._process_state_message(msg, topic)
                    current_state.update(state_data)
                
                # Process action commands
                elif topic in self.topic_config['action_topics']:
                    current_action = self._process_action_message(msg, topic)
                
                # Process language annotations
                elif topic in self.topic_config.get('language_topics', []):
                    language_annotation = self._process_language_message(msg)
                    episode_data['language'].append({
                        'timestamp': timestamp,
                        'text': language_annotation
                    })
                
                # Create observation when we have sufficient data
                if self._is_complete_observation(current_state, current_images, current_action):
                    observation = {
                        'state': current_state.copy(),
                        'images': current_images.copy(),
                        'timestamp': timestamp
                    }
                    
                    episode_data['observations'].append(observation)
                    episode_data['timestamps'].append(timestamp)
                    
                    if current_action is not None:
                        episode_data['actions'].append(current_action.copy())
                    
                    # Store video frames
                    for camera_name, image in current_images.items():
                        episode_data['videos'][camera_name].append(image)
        
        # Convert video lists to numpy arrays
        for camera_name in episode_data['videos']:
            if episode_data['videos'][camera_name]:
                episode_data['videos'][camera_name] = np.array(episode_data['videos'][camera_name])
        
        return episode_data
    
    def _topic_to_camera_name(self, topic: str) -> str:
        """Convert ROS topic name to camera name."""
        # Extract camera name from topic
        # e.g., "/camera/rgb/image_raw" -> "rgb"
        parts = topic.strip('/').split('/')
        if len(parts) >= 2:
            return parts[1]  # Usually the camera identifier
        return "camera"
    
    def _process_image_message(self, msg: Image) -> np.ndarray:
        """Process a ROS Image message."""
        try:
            # Convert ROS image to OpenCV format
            cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            return rgb_image
        except Exception as e:
            self.logger.error(f"Failed to process image message: {e}")
            # Return dummy image
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def _process_state_message(self, msg: Any, topic: str) -> Dict[str, np.ndarray]:
        """Process a state message (JointState, pose, etc.)."""
        state_data = {}
        
        if hasattr(msg, 'position') and hasattr(msg, 'name'):
            # JointState message
            for i, joint_name in enumerate(msg.name):
                if i < len(msg.position):
                    state_data[f"joint_{joint_name}"] = np.array([msg.position[i]])
            
            if hasattr(msg, 'velocity') and msg.velocity:
                for i, joint_name in enumerate(msg.name):
                    if i < len(msg.velocity):
                        state_data[f"joint_vel_{joint_name}"] = np.array([msg.velocity[i]])
        
        elif hasattr(msg, 'pose'):
            # PoseStamped message
            pose = msg.pose if hasattr(msg, 'pose') else msg
            pos = pose.position
            ori = pose.orientation
            
            state_data['ee_position'] = np.array([pos.x, pos.y, pos.z])
            state_data['ee_orientation'] = np.array([ori.x, ori.y, ori.z, ori.w])
        
        elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
            # Twist message
            state_data['linear_velocity'] = np.array([msg.linear.x, msg.linear.y, msg.linear.z])
            state_data['angular_velocity'] = np.array([msg.angular.x, msg.angular.y, msg.angular.z])
        
        return state_data
    
    def _process_action_message(self, msg: Any, topic: str) -> Optional[np.ndarray]:
        """Process an action message."""
        if hasattr(msg, 'position') and hasattr(msg, 'name'):
            # JointState-like action
            return np.array(msg.position)
        
        elif hasattr(msg, 'pose'):
            # Pose action
            pose = msg.pose if hasattr(msg, 'pose') else msg
            pos = pose.position
            ori = pose.orientation
            return np.array([pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w])
        
        elif hasattr(msg, 'linear') and hasattr(msg, 'angular'):
            # Twist action
            return np.array([
                msg.linear.x, msg.linear.y, msg.linear.z,
                msg.angular.x, msg.angular.y, msg.angular.z
            ])
        
        elif hasattr(msg, 'data') and isinstance(msg.data, (list, tuple)):
            # Generic array message
            return np.array(msg.data)
        
        return None
    
    def _process_language_message(self, msg: String) -> str:
        """Process a language annotation message."""
        return msg.data
    
    def _is_complete_observation(self, state: Dict, images: Dict, action: Any) -> bool:
        """Check if we have a complete observation."""
        # Require at least some state data and images
        has_state = len(state) > 0
        has_images = len(images) > 0
        
        # For the first few observations, action might be None
        return has_state and has_images
    
    def _extract_states_and_actions(self, episode_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract state and action arrays from episode data.
        
        Args:
            episode_data: Episode data from _load_episode_data
            
        Returns:
            Tuple of (states, actions) as numpy arrays
        """
        states_list = []
        actions_list = []
        
        # Extract states from observations
        for obs in episode_data['observations']:
            state_vector = self._concatenate_state_vector(obs['state'])
            states_list.append(state_vector)
        
        # Extract actions - pad if necessary
        actions_list = episode_data['actions']
        
        # Ensure we have the same number of states and actions
        if len(actions_list) < len(states_list):
            # Pad actions with the last action or zeros
            last_action = actions_list[-1] if actions_list else np.zeros(
                sum(self.embodiment_config['action_keys'].values())
            )
            while len(actions_list) < len(states_list):
                actions_list.append(last_action.copy())
        
        elif len(actions_list) > len(states_list):
            # Truncate actions to match states
            actions_list = actions_list[:len(states_list)]
        
        states = np.array(states_list)
        actions = np.array(actions_list)
        
        return states, actions
    
    def _concatenate_state_vector(self, state_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Concatenate state components into a single vector."""
        state_vector = []
        
        # Follow the order defined in embodiment_config
        for key in self.embodiment_config['state_keys']:
            if key in state_dict:
                state_component = state_dict[key]
                if isinstance(state_component, (list, tuple)):
                    state_component = np.array(state_component)
                state_vector.extend(state_component.flatten())
            else:
                # Fill with zeros if component is missing
                size = self.embodiment_config['state_keys'][key]
                state_vector.extend([0.0] * size)
        
        return np.array(state_vector, dtype=np.float32)


# Example usage and configuration helpers
def create_humanoid_config() -> Dict[str, Any]:
    """Create a default configuration for humanoid robots."""
    return {
        'state_keys': {
            'left_arm': 7,
            'left_hand': 6, 
            'right_arm': 7,
            'right_hand': 6,
            'torso': 3,
            'head': 2,
            'base_position': 3,
            'base_orientation': 4
        },
        'action_keys': {
            'left_arm': 7,
            'left_hand': 6,
            'right_arm': 7, 
            'right_hand': 6,
            'torso': 3,
            'head': 2
        },
        'camera_names': ['head_camera', 'chest_camera', 'hand_camera'],
        'embodiment_tag': 'gr1'
    }


def create_manipulation_config() -> Dict[str, Any]:
    """Create a default configuration for manipulation robots."""
    return {
        'state_keys': {
            'arm_joints': 7,
            'gripper': 2,
            'ee_position': 3,
            'ee_orientation': 4
        },
        'action_keys': {
            'arm_joints': 7,
            'gripper': 2
        },
        'camera_names': ['wrist_camera', 'shoulder_camera'],
        'embodiment_tag': 'oxe_droid'
    }


def create_topic_config() -> Dict[str, Any]:
    """Create a default ROS topic configuration."""
    return {
        'camera_topics': [
            '/camera/rgb/image_raw',
            '/head_camera/image_raw',
            '/wrist_camera/image_raw'
        ],
        'state_topics': [
            '/joint_states',
            '/robot_state',
            '/ee_pose'
        ],
        'action_topics': [
            '/joint_command',
            '/cartesian_command',
            '/gripper_command'
        ],
        'language_topics': [
            '/task_description',
            '/language_instruction'
        ]
    }

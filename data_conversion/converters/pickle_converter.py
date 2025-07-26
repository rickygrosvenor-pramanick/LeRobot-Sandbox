"""
Pickle converter for transforming pickled robot data to LeRobot format.

This converter processes pickle files containing robot demonstration data
and converts them to the LeRobot-compatible schema for Isaac-GR00T fine-tuning.
"""

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

from ..base_converter import BaseConverter


class PickleConverter(BaseConverter):
    """
    Converter for pickle files to LeRobot format.
    
    This converter assumes each pickle file contains a list of dictionaries,
    where each dictionary represents a single timestep with keys like:
    - 'observation': {'state': ..., 'image': ...}
    - 'action': ...
    - 'timestamp': ...
    """
    
    def _get_episode_paths(self, input_path: Path) -> List[Path]:
        """Get list of pickle files from input directory."""
        pickle_files = list(input_path.glob("*.pkl"))
        if not pickle_files:
            pickle_files = list(input_path.glob("*.pickle"))
        if not pickle_files:
            # Try recursive search
            pickle_files = list(input_path.glob("**/*.pkl"))
            pickle_files.extend(list(input_path.glob("**/*.pickle")))
        return sorted(pickle_files)
    
    def _load_episode_data(self, episode_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from a pickle file.
        
        Args:
            episode_path: Path to the .pkl file
            
        Returns:
            Dictionary containing episode data
        """
        file_path = Path(episode_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Pickle file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            raw_episode_data = pickle.load(f)
        
        # Process raw data into a standardized format
        episode_data = {
            'observations': [],
            'actions': [],
            'timestamps': [],
            'videos': {},
            'language': []
        }
        
        # Initialize video storage
        for camera_name in self.embodiment_config.get('camera_names', ['ego_view']):
            episode_data['videos'][camera_name] = []
        
        if not isinstance(raw_episode_data, list):
            raise TypeError("Pickle file should contain a list of timestep dictionaries.")
            
        for timestep_data in raw_episode_data:
            if not isinstance(timestep_data, dict):
                self.logger.warning("Skipping non-dictionary timestep data.")
                continue
            
            # Extract observation
            obs = timestep_data.get('observation', {})
            state = obs.get('state', np.array([]))
            
            # Extract images
            images = {}
            if 'image' in obs:
                # Single camera case
                images['ego_view'] = obs['image']
            elif 'images' in obs and isinstance(obs['images'], dict):
                # Multi-camera case
                images = obs['images']
            
            # Extract action
            action = timestep_data.get('action', np.array([]))
            
            # Extract timestamp
            timestamp = timestep_data.get('timestamp', len(episode_data['timestamps']) / self.video_fps)
            
            # Store data
            episode_data['observations'].append({'state': state, 'images': images, 'timestamp': timestamp})
            episode_data['actions'].append(action)
            episode_data['timestamps'].append(timestamp)
            
            for camera_name, image_data in images.items():
                if camera_name not in episode_data['videos']:
                    episode_data['videos'][camera_name] = []
                episode_data['videos'][camera_name].append(image_data)
        
        # Convert video lists to numpy arrays
        for camera_name in episode_data['videos']:
            if episode_data['videos'][camera_name]:
                episode_data['videos'][camera_name] = np.array(episode_data['videos'][camera_name])
        
        return episode_data
    
    def _extract_states_and_actions(self, episode_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract state and action arrays from episode data.
        
        Args:
            episode_data: Episode data from _load_episode_data
            
        Returns:
            Tuple of (states, actions) as numpy arrays
        """
        states_list = [obs['state'] for obs in episode_data['observations']]
        actions_list = episode_data['actions']
        
        # Pad or truncate actions to match states
        if len(actions_list) < len(states_list):
            last_action = actions_list[-1] if actions_list else np.zeros(
                sum(self.embodiment_config['action_keys'].values())
            )
            actions_list.extend([last_action] * (len(states_list) - len(actions_list)))
        elif len(actions_list) > len(states_list):
            actions_list = actions_list[:len(states_list)]
            
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(actions_list, dtype=np.float32)
        
        return states, actions

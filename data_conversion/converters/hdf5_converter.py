"""
HDF5 converter for transforming HDF5 robot data to LeRobot format.

This converter processes HDF5 files containing robot demonstration data
and converts them to the LeRobot-compatible schema for Isaac-GR00T fine-tuning.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    HDF5_AVAILABLE = False

from ..base_converter import BaseConverter


class HDF5Converter(BaseConverter):
    """
    Converter for HDF5 files to LeRobot format.
    
    This converter processes HDF5 files with a typical structure:
    - /observations/images/{camera_name}
    - /observations/state
    - /actions
    - /language (optional)
    """
    
    def __init__(self,
                 embodiment_config: Dict[str, Any],
                 hdf5_structure: Dict[str, str],
                 video_fps: int = 30,
                 chunk_size: int = 1000):
        """
        Initialize the HDF5 converter.
        
        Args:
            embodiment_config: Robot embodiment configuration
            hdf5_structure: Mapping of data types to HDF5 paths
            video_fps: Target frame rate for video data
            chunk_size: Number of episodes per chunk
        """
        if not HDF5_AVAILABLE:
            raise ImportError("h5py package is required for HDF5Converter")
            
        super().__init__(embodiment_config, video_fps, chunk_size)
        self.hdf5_structure = hdf5_structure
        
        # Default structure if not provided
        if not self.hdf5_structure:
            self.hdf5_structure = {
                'observations_state': '/observations/state',
                'actions': '/actions',
                'images_base': '/observations/images',
                'language': '/language'
            }
    
    def _get_episode_paths(self, input_path: Path) -> List[Path]:
        """Get list of HDF5 files from input directory."""
        hdf5_files = list(input_path.glob("*.hdf5"))
        if not hdf5_files:
            hdf5_files = list(input_path.glob("*.h5"))
        if not hdf5_files:
            # Try recursive search
            hdf5_files = list(input_path.glob("**/*.hdf5"))
            hdf5_files.extend(list(input_path.glob("**/*.h5")))
        return sorted(hdf5_files)
    
    def _load_episode_data(self, episode_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data from an HDF5 file.
        
        Args:
            episode_path: Path to the .hdf5/.h5 file
            
        Returns:
            Dictionary containing episode data
        """
        file_path = Path(episode_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {file_path}")
        
        episode_data = {
            'observations': [],
            'actions': [],
            'timestamps': [],
            'videos': {},
            'language': []
        }
        
        with h5py.File(file_path, 'r') as f:
            # Load states
            states_path = self.hdf5_structure.get('observations_state', '/observations/state')
            if states_path in f:
                states = np.array(f[states_path])
            else:
                self.logger.warning(f"States not found at {states_path}")
                states = np.array([])
            
            # Load actions
            actions_path = self.hdf5_structure.get('actions', '/actions')
            if actions_path in f:
                actions = np.array(f[actions_path])
            else:
                self.logger.warning(f"Actions not found at {actions_path}")
                actions = np.array([])
            
            # Load timestamps (if available)
            timestamps_path = self.hdf5_structure.get('timestamps', '/timestamps')
            if timestamps_path in f:
                timestamps = np.array(f[timestamps_path])
            else:
                # Generate timestamps based on video fps
                num_steps = len(states) if len(states) > 0 else len(actions)
                timestamps = np.arange(num_steps) / self.video_fps
            
            # Load videos
            images_base = self.hdf5_structure.get('images_base', '/observations/images')
            if images_base in f:
                images_group = f[images_base]
                for camera_name in images_group.keys():
                    camera_data = np.array(images_group[camera_name])
                    episode_data['videos'][camera_name] = camera_data
            
            # Load language annotations (if available)
            language_path = self.hdf5_structure.get('language', '/language')
            if language_path in f:
                language_data = f[language_path]
                if isinstance(language_data, h5py.Dataset):
                    # Single language annotation
                    language_text = language_data[()].decode('utf-8') if hasattr(language_data[()], 'decode') else str(language_data[()])
                    episode_data['language'] = [language_text]
                else:
                    # Multiple language annotations
                    episode_data['language'] = [
                        item.decode('utf-8') if hasattr(item, 'decode') else str(item)
                        for item in language_data
                    ]
        
        # Create observations from loaded data
        num_steps = len(states) if len(states) > 0 else len(actions)
        
        for i in range(num_steps):
            observation = {
                'state': states[i] if len(states) > i else np.array([]),
                'images': {},
                'timestamp': timestamps[i] if len(timestamps) > i else i / self.video_fps
            }
            
            # Add images for this timestep
            for camera_name, video_data in episode_data['videos'].items():
                if i < len(video_data):
                    observation['images'][camera_name] = video_data[i]
            
            episode_data['observations'].append(observation)
            episode_data['timestamps'].append(observation['timestamp'])
        
        # Store actions
        episode_data['actions'] = actions.tolist() if len(actions) > 0 else []
        
        return episode_data
    
    def _extract_states_and_actions(self, episode_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract state and action arrays from episode data.
        
        Args:
            episode_data: Episode data from _load_episode_data
            
        Returns:
            Tuple of (states, actions) as numpy arrays
        """
        states_list = []
        actions_list = episode_data['actions']
        
        # Extract states from observations
        for obs in episode_data['observations']:
            state = obs['state']
            if isinstance(state, (list, tuple)):
                state = np.array(state)
            elif not isinstance(state, np.ndarray):
                state = np.array([state])
            
            # Ensure state vector matches expected size
            expected_state_size = sum(self.embodiment_config['state_keys'].values())
            if len(state) != expected_state_size:
                if len(state) < expected_state_size:
                    # Pad with zeros
                    padded_state = np.zeros(expected_state_size)
                    padded_state[:len(state)] = state
                    state = padded_state
                else:
                    # Truncate
                    state = state[:expected_state_size]
            
            states_list.append(state)
        
        # Ensure actions match expected format
        expected_action_size = sum(self.embodiment_config['action_keys'].values())
        processed_actions = []
        
        for action in actions_list:
            if isinstance(action, (list, tuple)):
                action = np.array(action)
            elif not isinstance(action, np.ndarray):
                action = np.array([action])
            
            if len(action) != expected_action_size:
                if len(action) < expected_action_size:
                    # Pad with zeros
                    padded_action = np.zeros(expected_action_size)
                    padded_action[:len(action)] = action
                    action = padded_action
                else:
                    # Truncate
                    action = action[:expected_action_size]
            
            processed_actions.append(action)
        
        # Ensure we have the same number of states and actions
        if len(processed_actions) < len(states_list):
            # Pad actions with the last action or zeros
            last_action = processed_actions[-1] if processed_actions else np.zeros(expected_action_size)
            while len(processed_actions) < len(states_list):
                processed_actions.append(last_action.copy())
        
        elif len(processed_actions) > len(states_list):
            # Truncate actions to match states
            processed_actions = processed_actions[:len(states_list)]
        
        states = np.array(states_list, dtype=np.float32)
        actions = np.array(processed_actions, dtype=np.float32)
        
        return states, actions


def create_robomimic_structure() -> Dict[str, str]:
    """Create HDF5 structure for robomimic-style datasets."""
    return {
        'observations_state': '/data/obs/robot0_eef_pos',  # or joint states
        'actions': '/data/actions',
        'images_base': '/data/obs',
        'language': '/data/language'
    }


def create_rlds_structure() -> Dict[str, str]:
    """Create HDF5 structure for RLDS-style datasets."""
    return {
        'observations_state': '/steps/observation/state',
        'actions': '/steps/action',
        'images_base': '/steps/observation/image',
        'language': '/steps/language_instruction'
    }


def create_d4rl_structure() -> Dict[str, str]:
    """Create HDF5 structure for D4RL-style datasets."""
    return {
        'observations_state': '/observations',
        'actions': '/actions',
        'images_base': '/images',
        'timestamps': '/timestamps'
    }

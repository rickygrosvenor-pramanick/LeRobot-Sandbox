"""
Base converter class for transforming robot data to LeRobot format.

This module provides the foundational structure for converting various
robot data formats into the LeRobot-compatible schema required by Isaac-GR00T.
"""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm


class BaseConverter(ABC):
    """
    Abstract base class for robot data converters.
    
    This class defines the interface for converting robot demonstration data
    from various formats (ROS bags, HDF5, pickle, etc.) to the LeRobot format
    required by Isaac-GR00T.
    """
    
    def __init__(self, 
                 embodiment_config: Dict[str, Any],
                 video_fps: int = 30,
                 chunk_size: int = 1000):
        """
        Initialize the base converter.
        
        Args:
            embodiment_config: Configuration defining robot embodiment specifics
            video_fps: Frame rate for video data
            chunk_size: Number of episodes per chunk
        """
        self.embodiment_config = embodiment_config
        self.video_fps = video_fps
        self.chunk_size = chunk_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Validate embodiment configuration
        self._validate_embodiment_config()
    
    def _validate_embodiment_config(self) -> None:
        """Validate the embodiment configuration."""
        required_keys = ['state_keys', 'action_keys', 'embodiment_tag']
        for key in required_keys:
            if key not in self.embodiment_config:
                raise ValueError(f"Missing required key '{key}' in embodiment_config")
    
    @abstractmethod
    def _load_episode_data(self, episode_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load data for a single episode from the source format.
        
        Args:
            episode_path: Path to the episode data
            
        Returns:
            Dictionary containing episode data with keys:
            - 'observations': List of observation dictionaries
            - 'actions': List of action arrays
            - 'timestamps': List of timestamps
            - 'videos': Dictionary of video arrays per camera
            - 'language': Optional language annotations
        """
        pass
    
    @abstractmethod
    def _extract_states_and_actions(self, episode_data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract state and action arrays from episode data.
        
        Args:
            episode_data: Raw episode data
            
        Returns:
            Tuple of (states, actions) as numpy arrays
        """
        pass
    
    def convert(self,
                input_path: Union[str, Path],
                output_path: Union[str, Path],
                language_annotations: Optional[Dict[str, str]] = None,
                validation_split: float = 0.1) -> None:
        """
        Convert robot data to LeRobot format.
        
        Args:
            input_path: Path to input data directory
            output_path: Path to output LeRobot dataset directory
            language_annotations: Optional mapping of episode_id to task description
            validation_split: Fraction of data to reserve for validation
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        
        self.logger.info(f"Converting data from {input_path} to {output_path}")
        
        # Create output directory structure
        self._create_directory_structure(output_path)
        
        # Get list of episodes
        episode_paths = self._get_episode_paths(input_path)
        
        if not episode_paths:
            raise ValueError(f"No episodes found in {input_path}")
        
        # Split episodes for train/validation if needed
        train_episodes, val_episodes = self._split_episodes(episode_paths, validation_split)
        
        # Convert episodes
        all_episodes_info = []
        all_tasks = []
        
        for chunk_idx, episode_chunk in enumerate(self._chunk_episodes(train_episodes)):
            chunk_info = self._convert_episode_chunk(
                episode_chunk, output_path, chunk_idx, language_annotations
            )
            all_episodes_info.extend(chunk_info['episodes'])
            all_tasks.extend(chunk_info['tasks'])
        
        # Generate metadata files
        self._generate_metadata_files(output_path, all_episodes_info, all_tasks)
        
        self.logger.info("Conversion completed successfully!")
    
    def _create_directory_structure(self, output_path: Path) -> None:
        """Create the required LeRobot directory structure."""
        directories = [
            output_path / "meta",
            output_path / "data" / "chunk-000",
            output_path / "videos" / "chunk-000"
        ]
        
        for camera_name in self.embodiment_config.get('camera_names', ['ego_view']):
            video_dir = output_path / "videos" / "chunk-000" / f"observation.images.{camera_name}"
            directories.append(video_dir)
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _get_episode_paths(self, input_path: Path) -> List[Path]:
        """Get list of episode paths from input directory."""
        # This should be implemented by subclasses based on their data format
        # For now, return a generic implementation
        episode_paths = []
        
        # Look for common episode file patterns
        patterns = ['*.bag', '*.hdf5', '*.pkl', '*.json']
        for pattern in patterns:
            episode_paths.extend(input_path.glob(pattern))
        
        return sorted(episode_paths)
    
    def _split_episodes(self, episode_paths: List[Path], validation_split: float) -> Tuple[List[Path], List[Path]]:
        """Split episodes into train and validation sets."""
        num_episodes = len(episode_paths)
        num_val = int(num_episodes * validation_split)
        
        train_episodes = episode_paths[:-num_val] if num_val > 0 else episode_paths
        val_episodes = episode_paths[-num_val:] if num_val > 0 else []
        
        return train_episodes, val_episodes
    
    def _chunk_episodes(self, episode_paths: List[Path]) -> List[List[Path]]:
        """Group episodes into chunks."""
        chunks = []
        for i in range(0, len(episode_paths), self.chunk_size):
            chunks.append(episode_paths[i:i + self.chunk_size])
        return chunks
    
    def _convert_episode_chunk(self,
                              episode_paths: List[Path],
                              output_path: Path,
                              chunk_idx: int,
                              language_annotations: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Convert a chunk of episodes to LeRobot format."""
        chunk_episodes_info = []
        chunk_tasks = []
        
        for episode_idx, episode_path in enumerate(tqdm(episode_paths, desc=f"Converting chunk {chunk_idx}")):
            try:
                # Load episode data
                episode_data = self._load_episode_data(episode_path)
                
                # Extract states and actions
                states, actions = self._extract_states_and_actions(episode_data)
                
                # Convert to parquet format
                parquet_data = self._create_parquet_data(
                    episode_data, states, actions, episode_idx, language_annotations
                )
                
                # Save parquet file
                parquet_path = output_path / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{episode_idx:06d}.parquet"
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                parquet_data.to_parquet(parquet_path)
                
                # Save videos
                self._save_episode_videos(episode_data, output_path, chunk_idx, episode_idx)
                
                # Record episode info
                episode_info = {
                    "episode_index": episode_idx,
                    "length": len(states),
                    "tasks": [0]  # Default task index
                }
                chunk_episodes_info.append(episode_info)
                
                # Record task info
                task_description = self._get_task_description(episode_path, language_annotations)
                task_info = {
                    "task_index": len(chunk_tasks),
                    "task": task_description
                }
                chunk_tasks.append(task_info)
                
            except Exception as e:
                self.logger.error(f"Failed to convert episode {episode_path}: {e}")
                continue
        
        return {
            'episodes': chunk_episodes_info,
            'tasks': chunk_tasks
        }
    
    def _create_parquet_data(self,
                           episode_data: Dict[str, Any],
                           states: np.ndarray,
                           actions: np.ndarray,
                           episode_idx: int,
                           language_annotations: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """Create parquet data for an episode."""
        num_steps = len(states)
        
        # Basic data structure
        data = {
            'observation.state': [state.tolist() for state in states],
            'action': [action.tolist() for action in actions],
            'timestamp': episode_data.get('timestamps', np.arange(num_steps) * (1.0 / self.video_fps)),
            'episode_index': [episode_idx] * num_steps,
            'index': list(range(num_steps)),
            'next.reward': [0.0] * num_steps,
            'next.done': [False] * (num_steps - 1) + [True]
        }
        
        # Add language annotations if available
        task_description = self._get_task_description(episode_idx, language_annotations)
        if task_description:
            data['annotation.human.action.task_description'] = [0] * num_steps
            data['task_index'] = [0] * num_steps
            data['annotation.human.validity'] = [1] * num_steps  # Assume valid by default
        
        return pd.DataFrame(data)
    
    def _save_episode_videos(self,
                           episode_data: Dict[str, Any],
                           output_path: Path,
                           chunk_idx: int,
                           episode_idx: int) -> None:
        """Save video data for an episode."""
        videos = episode_data.get('videos', {})
        
        for camera_name, video_frames in videos.items():
            video_dir = output_path / "videos" / f"chunk-{chunk_idx:03d}" / f"observation.images.{camera_name}"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
            self._save_video_frames(video_frames, video_path)
    
    def _save_video_frames(self, frames: np.ndarray, output_path: Path) -> None:
        """Save video frames as MP4 file."""
        try:
            import cv2
            
            if frames.ndim == 4:  # (T, H, W, C)
                height, width = frames.shape[1:3]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(str(output_path), fourcc, self.video_fps, (width, height))
                
                for frame in frames:
                    if frame.shape[-1] == 3:  # RGB
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame.astype(np.uint8))
                
                out.release()
            else:
                self.logger.warning(f"Unexpected video shape: {frames.shape}")
                
        except ImportError:
            self.logger.error("OpenCV not available for video saving")
        except Exception as e:
            self.logger.error(f"Failed to save video {output_path}: {e}")
    
    def _get_task_description(self, episode_identifier: Any, language_annotations: Optional[Dict[str, str]]) -> str:
        """Get task description for an episode."""
        if language_annotations and str(episode_identifier) in language_annotations:
            return language_annotations[str(episode_identifier)]
        return "Robot demonstration task"  # Default description
    
    def _generate_metadata_files(self,
                                output_path: Path,
                                episodes_info: List[Dict[str, Any]],
                                tasks_info: List[Dict[str, Any]]) -> None:
        """Generate required metadata files for LeRobot format."""
        meta_dir = output_path / "meta"
        
        # Generate episodes.jsonl
        with open(meta_dir / "episodes.jsonl", 'w') as f:
            for episode in episodes_info:
                f.write(json.dumps(episode) + '\n')
        
        # Generate tasks.jsonl
        with open(meta_dir / "tasks.jsonl", 'w') as f:
            for task in tasks_info:
                f.write(json.dumps(task) + '\n')
        
        # Generate modality.json
        modality_config = self._create_modality_config()
        with open(meta_dir / "modality.json", 'w') as f:
            json.dump(modality_config, f, indent=2)
        
        # Generate info.json
        info_config = self._create_info_config()
        with open(meta_dir / "info.json", 'w') as f:
            json.dump(info_config, f, indent=2)
    
    def _create_modality_config(self) -> Dict[str, Any]:
        """Create the modality.json configuration."""
        state_config = {}
        action_config = {}
        
        # Build state configuration
        current_idx = 0
        for key, size in self.embodiment_config['state_keys'].items():
            state_config[key] = {
                "start": current_idx,
                "end": current_idx + size
            }
            current_idx += size
        
        # Build action configuration
        current_idx = 0
        for key, size in self.embodiment_config['action_keys'].items():
            action_config[key] = {
                "start": current_idx,
                "end": current_idx + size,
                "absolute": True  # Default to absolute actions
            }
            current_idx += size
        
        # Video configuration
        video_config = {}
        for camera_name in self.embodiment_config.get('camera_names', ['ego_view']):
            video_config[camera_name] = {
                "original_key": f"observation.images.{camera_name}"
            }
        
        # Annotation configuration
        annotation_config = {
            "human.action.task_description": {},
            "human.validity": {}
        }
        
        return {
            "state": state_config,
            "action": action_config,
            "video": video_config,
            "annotation": annotation_config
        }
    
    def _create_info_config(self) -> Dict[str, Any]:
        """Create the info.json configuration."""
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": [sum(self.embodiment_config['state_keys'].values())],
                "names": ["dim_0"]
            },
            "action": {
                "dtype": "float32", 
                "shape": [sum(self.embodiment_config['action_keys'].values())],
                "names": ["dim_0"]
            }
        }
        
        # Add video features
        for camera_name in self.embodiment_config.get('camera_names', ['ego_view']):
            video_key = f"observation.images.{camera_name}"
            features[video_key] = {
                "shape": [480, 640, 3],  # Default resolution
                "names": ["height", "width", "channel"],
                "info": {
                    "video.fps": self.video_fps,
                    "video.channels": 3
                }
            }
        
        return {
            "chunks_size": self.chunk_size,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": features
        }

"""
Dataset validation utility for Isaac-GR00T LeRobot-compatible datasets.

This script validates that a converted dataset meets the requirements
for Isaac-GR00T fine-tuning.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Any

import pandas as pd


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def validate_directory_structure(dataset_path: Path) -> bool:
    """Validate the directory structure of the dataset."""
    print("🔍 Validating directory structure...")
    
    required_dirs = [
        "meta",
        "data", 
        "videos"
    ]
    
    all_valid = True
    
    for dir_name in required_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists() and dir_path.is_dir():
            print(f"✅ Found directory: {dir_name}/")
        else:
            print(f"❌ Missing directory: {dir_name}/")
            all_valid = False
    
    return all_valid


def validate_metadata_files(dataset_path: Path) -> bool:
    """Validate required metadata files."""
    print("\n🔍 Validating metadata files...")
    
    required_files = [
        "meta/modality.json",
        "meta/tasks.jsonl",
        "meta/episodes.jsonl", 
        "meta/info.json"
    ]
    
    all_valid = True
    
    for file_path in required_files:
        full_path = dataset_path / file_path
        if full_path.exists() and full_path.is_file():
            print(f"✅ Found file: {file_path}")
            
            # Validate JSON format
            try:
                if file_path.endswith('.json'):
                    with open(full_path, 'r') as f:
                        json.load(f)
                elif file_path.endswith('.jsonl'):
                    with open(full_path, 'r') as f:
                        for line in f:
                            json.loads(line.strip())
                print(f"   ✅ Valid JSON format")
            except json.JSONDecodeError as e:
                print(f"   ❌ Invalid JSON format: {e}")
                all_valid = False
        else:
            print(f"❌ Missing file: {file_path}")
            all_valid = False
    
    return all_valid


def validate_modality_config(dataset_path: Path) -> bool:
    """Validate the modality.json configuration."""
    print("\n🔍 Validating modality configuration...")
    
    modality_path = dataset_path / "meta/modality.json"
    if not modality_path.exists():
        print("❌ modality.json not found")
        return False
    
    try:
        with open(modality_path, 'r') as f:
            modality_config = json.load(f)
        
        required_sections = ['state', 'action', 'video']
        
        for section in required_sections:
            if section not in modality_config:
                print(f"❌ Missing section: {section}")
                return False
            print(f"✅ Found section: {section}")
            
            # Validate state and action sections
            if section in ['state', 'action']:
                for key, config in modality_config[section].items():
                    if 'start' not in config or 'end' not in config:
                        print(f"   ❌ Missing start/end in {section}.{key}")
                        return False
                    print(f"   ✅ {section}.{key}: indices {config['start']}-{config['end']}")
            
            # Validate video section
            elif section == 'video':
                for key, config in modality_config[section].items():
                    original_key = config.get('original_key', key)
                    print(f"   ✅ {section}.{key}: maps to {original_key}")
        
        # Check annotation section if present
        if 'annotation' in modality_config:
            print("✅ Found annotation section")
            for key in modality_config['annotation']:
                print(f"   ✅ annotation.{key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating modality config: {e}")
        return False


def validate_data_files(dataset_path: Path) -> bool:
    """Validate data parquet files."""
    print("\n🔍 Validating data files...")
    
    data_dir = dataset_path / "data"
    if not data_dir.exists():
        print("❌ Data directory not found")
        return False
    
    # Find all parquet files
    parquet_files = list(data_dir.glob("**/*.parquet"))
    
    if not parquet_files:
        print("❌ No parquet files found")
        return False
    
    print(f"✅ Found {len(parquet_files)} parquet files")
    
    # Validate a sample parquet file
    sample_file = parquet_files[0]
    try:
        df = pd.read_parquet(sample_file)
        print(f"✅ Successfully loaded sample file: {sample_file.name}")
        print(f"   Shape: {df.shape}")
        
        required_columns = [
            'observation.state',
            'action',
            'timestamp',
            'episode_index',
            'index'
        ]
        
        for col in required_columns:
            if col in df.columns:
                print(f"   ✅ Found column: {col}")
            else:
                print(f"   ❌ Missing column: {col}")
                return False
        
        # Check data types
        state_data = df['observation.state'].iloc[0]
        action_data = df['action'].iloc[0]
        
        if isinstance(state_data, list) and isinstance(action_data, list):
            print(f"   ✅ State dimension: {len(state_data)}")
            print(f"   ✅ Action dimension: {len(action_data)}")
        else:
            print(f"   ❌ State/action data not in list format")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating data files: {e}")
        return False


def validate_video_files(dataset_path: Path) -> bool:
    """Validate video files."""
    print("\n🔍 Validating video files...")
    
    videos_dir = dataset_path / "videos"
    if not videos_dir.exists():
        print("❌ Videos directory not found")
        return False
    
    # Find all video files
    video_files = list(videos_dir.glob("**/*.mp4"))
    
    if not video_files:
        print("❌ No video files found")
        return False
    
    print(f"✅ Found {len(video_files)} video files")
    
    # Check video directory structure
    camera_dirs = [d for d in videos_dir.glob("*/observation.images.*") if d.is_dir()]
    
    if camera_dirs:
        print("✅ Found camera directories:")
        for cam_dir in camera_dirs:
            camera_name = cam_dir.name.replace("observation.images.", "")
            videos_in_dir = list(cam_dir.glob("*.mp4"))
            print(f"   📹 {camera_name}: {len(videos_in_dir)} videos")
    else:
        print("⚠️  No properly structured camera directories found")
    
    return True


def validate_dataset_consistency(dataset_path: Path) -> bool:
    """Validate consistency between metadata and data files."""
    print("\n🔍 Validating dataset consistency...")
    
    try:
        # Load episodes metadata
        episodes_path = dataset_path / "meta/episodes.jsonl"
        episodes = []
        with open(episodes_path, 'r') as f:
            for line in f:
                episodes.append(json.loads(line.strip()))
        
        print(f"✅ Metadata contains {len(episodes)} episodes")
        
        # Check if data files exist for each episode
        data_dir = dataset_path / "data"
        missing_episodes = []
        
        for episode in episodes:
            episode_idx = episode['episode_index']
            episode_file = None
            
            # Look for episode file in chunks
            for chunk_dir in data_dir.glob("chunk-*"):
                potential_file = chunk_dir / f"episode_{episode_idx:06d}.parquet"
                if potential_file.exists():
                    episode_file = potential_file
                    break
            
            if episode_file is None:
                missing_episodes.append(episode_idx)
        
        if missing_episodes:
            print(f"❌ Missing data files for episodes: {missing_episodes[:5]}...")
            return False
        else:
            print("✅ All episodes have corresponding data files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating consistency: {e}")
        return False


def generate_dataset_report(dataset_path: Path) -> Dict[str, Any]:
    """Generate a comprehensive dataset report."""
    print("\n📊 Generating dataset report...")
    
    report = {
        'dataset_path': str(dataset_path),
        'validation_passed': True,
        'warnings': [],
        'errors': [],
        'statistics': {}
    }
    
    try:
        # Basic statistics
        data_dir = dataset_path / "data"
        parquet_files = list(data_dir.glob("**/*.parquet"))
        
        total_steps = 0
        total_episodes = len(parquet_files)
        
        for pfile in parquet_files[:10]:  # Sample first 10 files
            df = pd.read_parquet(pfile)
            total_steps += len(df)
        
        # Estimate total steps
        estimated_total_steps = (total_steps / min(10, len(parquet_files))) * len(parquet_files)
        
        videos_dir = dataset_path / "videos"
        video_files = list(videos_dir.glob("**/*.mp4")) if videos_dir.exists() else []
        
        report['statistics'] = {
            'total_episodes': total_episodes,
            'estimated_total_steps': int(estimated_total_steps),
            'total_video_files': len(video_files),
            'parquet_files': len(parquet_files)
        }
        
        # Load modality config for dimensions
        modality_path = dataset_path / "meta/modality.json"
        if modality_path.exists():
            with open(modality_path, 'r') as f:
                modality_config = json.load(f)
            
            state_dim = 0
            action_dim = 0
            
            for key, config in modality_config.get('state', {}).items():
                state_dim += config['end'] - config['start']
            
            for key, config in modality_config.get('action', {}).items():
                action_dim += config['end'] - config['start']
            
            report['statistics']['state_dimension'] = state_dim
            report['statistics']['action_dimension'] = action_dim
        
        print("✅ Dataset report generated")
        
    except Exception as e:
        print(f"⚠️  Error generating report: {e}")
        report['errors'].append(str(e))
    
    return report


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Isaac-GR00T LeRobot dataset")
    parser.add_argument(
        "--dataset-path", 
        type=str, 
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        help="Path to save validation report (optional)"
    )
    
    args = parser.parse_args()
    
    setup_logging()
    
    dataset_path = Path(args.dataset_path)
    
    print(f"🔍 Validating dataset: {dataset_path}")
    print("=" * 60)
    
    if not dataset_path.exists():
        print(f"❌ Dataset path does not exist: {dataset_path}")
        return
    
    # Run all validations
    validations = [
        validate_directory_structure(dataset_path),
        validate_metadata_files(dataset_path),
        validate_modality_config(dataset_path),
        validate_data_files(dataset_path),
        validate_video_files(dataset_path),
        validate_dataset_consistency(dataset_path)
    ]
    
    # Generate report
    report = generate_dataset_report(dataset_path)
    
    # Summary
    print("\n📋 Validation Summary")
    print("=" * 30)
    
    if all(validations):
        print("✅ All validations passed!")
        print("🚀 Dataset is ready for Isaac-GR00T fine-tuning")
        report['validation_passed'] = True
    else:
        print("❌ Some validations failed")
        print("🔧 Please fix the issues before proceeding")
        report['validation_passed'] = False
    
    # Print statistics
    if 'statistics' in report:
        stats = report['statistics']
        print(f"\n📊 Dataset Statistics:")
        print(f"   Episodes: {stats.get('total_episodes', 'N/A')}")
        print(f"   Estimated steps: {stats.get('estimated_total_steps', 'N/A')}")
        print(f"   Video files: {stats.get('total_video_files', 'N/A')}")
        print(f"   State dimension: {stats.get('state_dimension', 'N/A')}")
        print(f"   Action dimension: {stats.get('action_dimension', 'N/A')}")
    
    # Save report if requested
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Validation report saved to: {report_path}")


if __name__ == "__main__":
    main()

import kagglehub
import shutil
import os
from pathlib import Path

def import_data(path: str = "data/raw_videos"):
    """
    Download the exercise recognition dataset and copy to specified path with same structure.
    
    Args:
        path: Local path where you want the raw videos stored
    
    Returns:
        str: Path where the videos are stored
    """
    try:
        
        print(f"Dataset downloaded to: {path}")
        
        # Create target directory
        target_path = Path(path)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # Copy the entire directory structure to your specified path
        if raw_videos_path != str(target_path):
            print(f"Copying files to: {target_path}")
            
            # Copy all contents maintaining directory structure
            for item in os.listdir(raw_videos_path):
                source = os.path.join(raw_videos_path, item)
                destination = target_path / item
                
                if os.path.isdir(source):
                    shutil.copytree(source, destination, dirs_exist_ok=True)
                else:
                    shutil.copy2(source, destination)
            
            print("Copy completed!")
            return str(target_path)
        
        return raw_videos_path
        
    except Exception as e:
        print(f"Error importing data: {e}")
        return None

def transform_data(input_path: str = "data/raw_videos", output_path: str = "data/processed"):
    """
    Placeholder for data transformation - currently just creates output directory.
    
    Args:
        input_path: Path to raw video files
        output_path: Path for processed data
    
    Returns:
        str: Output path for processed data
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Raw videos location: {input_path}")
    print(f"Processed data will go to: {output_path}")
    
    # List what's in the raw videos directory
    if os.path.exists(input_path):
        print("\nContents of raw videos directory:")
        for root, dirs, files in os.walk(input_path):
            level = root.replace(input_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    
    return output_path

# Example usage:
if __name__ == "__main__":
    # Import raw videos to your local directory
    path = kagglehub.dataset_download("riccardoriccio/real-time-exercise-recognition-dataset")
    # Set up processed data directory
    #processed_path = transform_data(video_path, "data/processed")
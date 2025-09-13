import mediapipe as mp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import kagglehub


class MediaPipeFeatureExtractor:
    """Extract pose coordinates using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_orientation_from_pose(self, landmarks):
        """
        Calculate body orientation (roll, pitch, yaw) from pose landmarks
        Returns 4D orientation: roll, pitch, yaw, confidence
        """
        try:
            # Key points for orientation calculation
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
            right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            
            # Calculate shoulder and hip midpoints
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            # Calculate orientation vectors
            # Torso vector (vertical axis)
            torso_vector = shoulder_center - hip_center
            torso_vector = torso_vector / (np.linalg.norm(torso_vector) + 1e-8)
            
            # Shoulder vector (horizontal axis)
            shoulder_vector = right_shoulder - left_shoulder
            shoulder_vector = shoulder_vector / (np.linalg.norm(shoulder_vector) + 1e-8)
            
            # Forward vector (perpendicular to both)
            forward_vector = np.cross(torso_vector, shoulder_vector)
            forward_vector = forward_vector / (np.linalg.norm(forward_vector) + 1e-8)
            
            # Create rotation matrix
            rotation_matrix = np.column_stack([shoulder_vector, forward_vector, torso_vector])
            
            # Convert to Euler angles (roll, pitch, yaw)
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)  # roll, pitch, yaw
            
            # Calculate confidence based on landmark visibility
            visibility_scores = [
                landmarks[11].visibility, landmarks[12].visibility,
                landmarks[23].visibility, landmarks[24].visibility
            ]
            confidence = np.mean(visibility_scores)
            
            return np.array([euler_angles[0], euler_angles[1], euler_angles[2], confidence])
            
        except Exception as e:
            # Return zero orientation with low confidence if calculation fails
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def extract_video_features(self, video_path):
        """Extract 3D coordinates (x,y,z) + orientation (roll,pitch,yaw,conf) per frame"""
        cap = cv2.VideoCapture(video_path)
        frame_features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract 3D coordinates for 33 keypoints = 99 features
                coords = []
                for landmark in results.pose_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                # Extract orientation (roll, pitch, yaw, confidence) = 4 features
                orientation = self.calculate_orientation_from_pose(results.pose_landmarks.landmark)
                
                # Combine: 99 + 4 = 103 features per frame
                frame_feature = coords + orientation.tolist()
                frame_features.append(frame_feature)
        
        cap.release()
        return np.array(frame_features)
    
    def process_dataset(self, dataset_path="exercise_dataset"):
        """Process all videos and create CSV"""
        dataset_path = Path(dataset_path)
        raw_videos_path = dataset_path / "raw_videos"
        
        all_data = []
        exercise_to_id = {}
        current_id = 0
        
        print("Processing dataset videos...")
        
        for exercise_folder in raw_videos_path.iterdir():
            if not exercise_folder.is_dir():
                continue
            
            exercise_name = exercise_folder.name
            if exercise_name not in exercise_to_id:
                exercise_to_id[exercise_name] = current_id
                current_id += 1
            
            print(f"Processing {exercise_name}...")
            
            video_files = list(exercise_folder.glob("*.mp4"))
            
            for i, video_file in enumerate(tqdm(video_files, desc=f"  {exercise_name} videos")):
                try:
                    # Extract features
                    features = self.extract_video_features(str(video_file))
                    
                    if len(features) == 0:
                        print(f"    Skipping {video_file.name}: No poses detected")
                        continue
                    
                    # Create video ID as paper describes
                    video_id = f"{exercise_name}_{i+1:03d}"
                    
                    # Add each frame as a row
                    for frame_idx, frame_coords in enumerate(features):
                        row = [video_id] + frame_coords.tolist() + [exercise_name]
                        all_data.append(row)
                
                except Exception as e:
                    print(f"    Error processing {video_file.name}: {e}")
        
        columns = ['video_id'] + [f'kp_{i//2}_{["x","y"][i%2]}' for i in range(66)] + ['class']
        df = pd.DataFrame(all_data, columns=columns)
        
        # Save CSV
        csv_path = dataset_path / "processed" / "pose_features.csv"
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(csv_path, index=False)
        
        print(f"Dataset processed: {len(df)} frames from {len(df['video_id'].unique())} videos")
        print(f"Exercise mapping: {exercise_to_id}")
        
        return df, exercise_to_id

class VideoGraphDataset:
    """Convert CSV data to graph format"""
    
    def __init__(self, csv_path, exercise_to_label):
        self.df = pd.read_csv(csv_path)
        self.exercise_to_label = exercise_to_label
        self.label_to_exercise = {v: k for k, v in exercise_to_label.items()}
        
        # Group by video_id to create graphs
        self.video_groups = self.df.groupby('video_id')
        self.video_ids = list(self.video_groups.groups.keys())
        
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        video_data = self.video_groups.get_group(video_id)
        
        # Extract node features (66 coordinates per frame)
        feature_cols = [col for col in self.df.columns if col.startswith('kp_')]
        node_features = video_data[feature_cols].values.astype(np.float32)
        
        # Create sequential edges (frame i -> frame i+1)
        num_frames = len(node_features)
        edge_index = []
        
        # Sequential connections between frames
        for i in range(num_frames - 1):
            edge_index.append([i, i + 1])
            edge_index.append([i + 1, i])  # Bidirectional
        
        # Convert to tensor format
        if len(edge_index) > 0:
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        else:
            # Single frame video - self connection
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
        
        exercise_name = video_data['class'].iloc[0]
        label = self.exercise_to_label[exercise_name]
        graph = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            y=torch.tensor(label, dtype=torch.long)
        )
        
        return graph

class ExerciseGNN(nn.Module):
    """Graph Neural Network classfier"""
    
    def __init__(self, num_features=66, hidden_dim=128, num_classes=3):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        # GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        # Global pooling (mean of all nodes in each graph)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            # Single graph case
            x = x.mean(dim=0, keepdim=True)
        
        # Classification
        x = self.classifier(x)
        
        return x

class ExerciseTrainer:
    """Training pipeline following paper's methodology"""
    
    def __init__(self, dataset_path="exercise_dataset"):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train(self):
        print("=== Exercise Recognition Training (Paper Method) ===\n")
        
        # Step 1: Extract features using MediaPipe
        csv_path = self.dataset_path / "processed" / "pose_features.csv"
        
        if not csv_path.exists():
            print("Step 1: Extracting pose features with MediaPipe...")
            extractor = MediaPipeFeatureExtractor()
            df, exercise_to_label = extractor.process_dataset(self.dataset_path)
        else:
            print("Step 1: Loading existing pose features...")
            df = pd.read_csv(csv_path)
            # Reconstruct exercise mapping
            exercise_names = df['class'].unique()
            exercise_to_label = {name: i for i, name in enumerate(exercise_names)}
        
        print(f"Features extracted: {len(df)} frames")
        print(f"Exercises: {list(exercise_to_label.keys())}")
        
        # Step 2: Create graph dataset
        print("\nStep 2: Creating graph dataset...")
        dataset = VideoGraphDataset(csv_path, exercise_to_label)
        print(f"Created {len(dataset)} video graphs")
        
        # Step 3: Train-test split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"Train videos: {len(train_dataset)}")
        print(f"Test videos: {len(test_dataset)}")
        
        # Step 4: Initialize model
        print(f"\nStep 3: Initializing GNN model...")
        model = ExerciseGNN(
            num_features=66,  # 33 keypoints * 2 (x,y)
            hidden_dim=128,
            num_classes=len(exercise_to_label)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training on: {self.device}")
        
        # Step 5: Training loop (following paper's hyperparameters)
        print("\nStep 4: Training model...")
        epochs = 200  # As in paper
        
        train_losses = []
        train_accuracies = []
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch in train_loader:
                batch = batch.to(self.device)
                optimizer.zero_grad()
                
                # Forward pass
                output = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(output, batch.y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += batch.y.size(0)
                correct += (predicted == batch.y).sum().item()
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            if epoch % 20 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch:3d}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # Step 6: Evaluation
        print("\nStep 5: Evaluating model...")
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(self.device)
                output = model(batch.x, batch.edge_index, batch.batch)
                _, predicted = torch.max(output, 1)
                
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        print(f"\n=== RESULTS ===")
        print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        # Classification report
        exercise_names = list(exercise_to_label.keys())
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=exercise_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=exercise_names, yticklabels=exercise_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save model
        model_path = self.dataset_path / "exercise_gnn_model.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'exercise_to_label': exercise_to_label,
            'test_accuracy': accuracy
        }, model_path)
        
        print(f"\nModel saved to {model_path}")
        print(f"Paper reproduction complete! Achieved {accuracy*100:.2f}% accuracy")
        
        return model, exercise_to_label, accuracy



if __name__ == "__main__":
    dataset_path = "../data/raw"
    print("=== MediaPipe + GNN Exercise Recognition ===")
    print("Following the exact methodology from the paper")
    print("'MediaPipe with GNN for Human Activity Recognition'")
    # Check for videos
    
    total_videos = 0
    for exercise_folder in raw_videos_path.iterdir():
        if exercise_folder.is_dir():
            video_count = len(list(exercise_folder.glob("*.mp4")))
            total_videos += video_count
            print(f"  {exercise_folder.name}: {video_count} videos")
    
    if total_videos > 0:
        print(f"\nFound {total_videos} total videos. Starting training...")
        trainer = ExerciseTrainer(dataset_path)
        model, exercise_mapping, accuracy = trainer.train()
        
        print(f"\n Training completed!")
        print(f" Final accuracy: {accuracy*100:.2f}%")

        print(f" Model saved for API usage")
        
    else:
        print(f"\n  No videos found. Please add MP4 files to the exercise folders.")
        print("Example: exercise_dataset/raw_videos/squats/squat_001.mp4")
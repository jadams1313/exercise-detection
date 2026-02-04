import mediapipe as mp
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


class MediaPipeFeatureExtractor:
    """Extract pose coordinates using MediaPipe - same as before"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def calculate_orientation_from_pose(self, landmarks):
        """Calculate body orientation (roll, pitch, yaw) from pose landmarks"""
        try:
            left_shoulder = np.array([landmarks[11].x, landmarks[11].y, landmarks[11].z])
            right_shoulder = np.array([landmarks[12].x, landmarks[12].y, landmarks[12].z])
            left_hip = np.array([landmarks[23].x, landmarks[23].y, landmarks[23].z])
            right_hip = np.array([landmarks[24].x, landmarks[24].y, landmarks[24].z])
            
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            
            torso_vector = shoulder_center - hip_center
            torso_vector = torso_vector / (np.linalg.norm(torso_vector) + 1e-8)
            
            shoulder_vector = right_shoulder - left_shoulder
            shoulder_vector = shoulder_vector / (np.linalg.norm(shoulder_vector) + 1e-8)
            
            forward_vector = np.cross(torso_vector, shoulder_vector)
            forward_vector = forward_vector / (np.linalg.norm(forward_vector) + 1e-8)
            
            rotation_matrix = np.column_stack([shoulder_vector, forward_vector, torso_vector])
            rotation = R.from_matrix(rotation_matrix)
            euler_angles = rotation.as_euler('xyz', degrees=True)
            
            visibility_scores = [
                landmarks[11].visibility, landmarks[12].visibility,
                landmarks[23].visibility, landmarks[24].visibility
            ]
            confidence = np.mean(visibility_scores)
            
            return np.array([euler_angles[0], euler_angles[1], euler_angles[2], confidence])
            
        except Exception as e:
            return np.array([0.0, 0.0, 0.0, 0.0])
    
    def extract_video_features(self, video_path):
        """Extract features per frame (no averaging)"""
        cap = cv2.VideoCapture(video_path)
        frame_features = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                # Extract 3D coordinates for 33 keypoints
                coords = []
                for landmark in results.pose_landmarks.landmark:
                    coords.extend([landmark.x, landmark.y, landmark.z])
                
                # Extract orientation
                orientation = self.calculate_orientation_from_pose(results.pose_landmarks.landmark)
                
                # 99 + 4 = 103 features per frame
                frame_feature = coords + orientation.tolist()
                frame_features.append(frame_feature)
        
        cap.release()
        return np.array(frame_features)
    
    def process_dataset(self, dataset_path):
        """Process all videos and create CSV - same as before"""
        dataset_path = Path(dataset_path)
        
        all_data = []
        exercise_to_id = {}
        current_id = 0
        
        print("Processing dataset videos...")
        
        for exercise_folder in dataset_path.iterdir():
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
                    features = self.extract_video_features(str(video_file))
                    
                    if len(features) == 0:
                        print(f"    Skipping {video_file.name}: No poses detected")
                        continue
                    
                    video_id = f"{exercise_name}_{i+1:03d}"
                    
                    for frame_idx, frame_coords in enumerate(features):
                        row = [video_id] + frame_coords.tolist() + [exercise_name]
                        all_data.append(row)
                
                except Exception as e:
                    print(f"    Error processing {video_file.name}: {e}")
        
        columns = ['video_id'] + [f'kp_{i//3}_{["x","y","z"][i%3]}' for i in range(99)] + \
                  ['orientation_roll', 'orientation_pitch', 'orientation_yaw', 'orientation_confidence'] + ['class']
        df = pd.DataFrame(all_data, columns=columns)
        
        csv_path = dataset_path / "processed" / "pose_features.csv"
        csv_path.parent.mkdir(exist_ok=True, parents=True)
        df.to_csv(csv_path, index=False)
        
        print(f"Dataset processed: {len(df)} frames from {len(df['video_id'].unique())} videos")
        print(f"Exercise mapping: {exercise_to_id}")
        
        return df, exercise_to_id


class TemporalWindowDataset(torch.utils.data.Dataset):
    """
    Dataset that creates temporal windows from pose sequences.
    Each sample is a window of N consecutive frames.
    """
    
    def __init__(self, csv_path, exercise_to_label, window_size=30, stride=15):
        """
        Args:
            csv_path: Path to pose features CSV
            exercise_to_label: Dictionary mapping exercise names to label indices
            window_size: Number of frames per window (e.g., 30 frames = 1 second at 30fps)
            stride: Number of frames to skip between windows (smaller = more overlap)
        """
        self.csv_path = csv_path
        self.exercise_to_label = exercise_to_label
        self.window_size = window_size
        self.stride = stride
        
        # Load the CSV data
        self.df = pd.read_csv(csv_path)
        self.video_ids = self.df['video_id'].unique()
        
        # Create windows from all videos
        self.windows = []
        self._create_windows()
        
        print(f"Created {len(self.windows)} windows from {len(self.video_ids)} videos")
        print(f"Window size: {window_size} frames, Stride: {stride} frames")
        
    def _create_windows(self):
        """Create sliding windows from each video"""
        for video_id in self.video_ids:
            # Get all frames for this video
            video_frames = self.df[self.df['video_id'] == video_id]
            
            # Extract features and label
            feature_cols = [col for col in self.df.columns if col not in ['video_id', 'class']]
            pose_features = video_frames[feature_cols].values.astype(np.float32)
            exercise_class = video_frames['class'].iloc[0]
            label = self.exercise_to_label[exercise_class]
            
            # Create sliding windows
            num_frames = len(pose_features)
            
            # If video is shorter than window_size, pad it
            if num_frames < self.window_size:
                # Pad by repeating the last frame
                padding = np.repeat(pose_features[-1:], self.window_size - num_frames, axis=0)
                pose_features = np.vstack([pose_features, padding])
                num_frames = self.window_size
            
            # Create windows with stride
            for start_idx in range(0, num_frames - self.window_size + 1, self.stride):
                end_idx = start_idx + self.window_size
                window = pose_features[start_idx:end_idx]
                self.windows.append((window, label, video_id))
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        window, label, video_id = self.windows[idx]
        
        # Convert window to sequence of graphs
        graphs = []
        for frame_idx in range(self.window_size):
            frame_features = window[frame_idx]
            node_features, edge_index = self.create_pose_graph(frame_features)
            
            # Create graph for this frame
            graph = Data(
                x=node_features,
                edge_index=edge_index
            )
            graphs.append(graph)
        
        return graphs, torch.tensor(label, dtype=torch.long), video_id
    
    def create_pose_graph(self, pose_features):
        """Convert single frame pose to graph"""
        # pose_features shape: [103]
        
        # Extract coordinates (first 99 features: 33 keypoints × 3)
        coords = pose_features[:99].reshape(33, 3)  # [keypoints, xyz]
        orientation = pose_features[99:]  # [4] - orientation features
        
        # Each node gets coordinates + orientation info
        orientation_repeated = np.tile(orientation, (33, 1))  # [33, 4]
        node_features = np.concatenate([coords, orientation_repeated], axis=1)  # [33, 7]
        
        # Create edges based on human pose skeleton
        edge_connections = [
            # Head connections
            (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
            # Torso
            (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
            # Left arm
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
            # Right arm
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
            # Left leg
            (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
            # Right leg
            (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
        ]
        
        # Convert to edge_index format (bidirectional)
        edges = []
        for src, dst in edge_connections:
            if src < 33 and dst < 33:
                edges.append([src, dst])
                edges.append([dst, src])
        
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        return torch.tensor(node_features, dtype=torch.float32), edge_index


class SpatialGNN(nn.Module):
    """
    STAGE 1: Spatial processing with GNN
    Processes each frame independently to extract pose embeddings
    """
    
    def __init__(self, num_features=7, hidden_dim=128, output_dim=64):
        super().__init__()
        
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x, edge_index, batch=None):
        """
        Args:
            x: Node features [num_nodes, 7]
            edge_index: Edge connections [2, num_edges]
            batch: Batch assignment for multiple graphs
        
        Returns:
            Graph embedding [batch_size, output_dim] or [1, output_dim] for single graph
        """
        # GCN layers with ReLU activation
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index)
        
        # Global pooling (mean of all nodes)
        if batch is not None:
            x = global_mean_pool(x, batch)
        else:
            x = x.mean(dim=0, keepdim=True)
        
        return x


class TemporalLSTM(nn.Module):
    """
    STAGE 2: Temporal processing with LSTM
    Takes sequence of GNN embeddings and learns temporal patterns
    """
    
    def __init__(self, input_dim=64, hidden_dim=128, num_layers=2, num_classes=5):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3 if num_layers > 1 else 0
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Sequence of graph embeddings [batch_size, sequence_length, embedding_dim]
        
        Returns:
            Class predictions [batch_size, num_classes]
        """
        # LSTM processes the sequence
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state for classification
        # lstm_out shape: [batch_size, seq_len, hidden_dim]
        # We take the last timestep
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_dim]
        
        # Classification
        output = self.classifier(last_output)
        
        return output


class TemporalExerciseModel(nn.Module):
    """
    Complete two-stage model: GNN + LSTM
    """
    
    def __init__(self, num_features=7, gnn_hidden=128, gnn_output=64, 
                 lstm_hidden=128, lstm_layers=2, num_classes=5):
        super().__init__()
        
        self.spatial_gnn = SpatialGNN(num_features, gnn_hidden, gnn_output)
        self.temporal_lstm = TemporalLSTM(gnn_output, lstm_hidden, lstm_layers, num_classes)
        
    def forward(self, graph_sequence):
        """
        Args:
            graph_sequence: List of graphs for one window
                           Each graph has x, edge_index
        
        Returns:
            Class predictions [batch_size, num_classes]
        """
        # STAGE 1: Process each graph with GNN to get embeddings
        embeddings = []
        for graph in graph_sequence:
            # Process single graph
            embedding = self.spatial_gnn(graph.x, graph.edge_index, graph.batch)
            embeddings.append(embedding)
        
        # Stack embeddings into sequence
        # embeddings: list of [batch_size, embedding_dim]
        sequence = torch.stack(embeddings, dim=1)  # [batch_size, seq_len, embedding_dim]
        
        # STAGE 2: Process sequence with LSTM
        output = self.temporal_lstm(sequence)
        
        return output


def collate_temporal_batch(batch):
    """
    Custom collate function to handle batches of graph sequences
    
    Args:
        batch: List of tuples (graphs, label, video_id)
               where graphs is a list of Data objects
    
    Returns:
        batched_sequences: List of batched graphs (one per timestep)
        labels: Tensor of labels
        video_ids: List of video IDs
    """
    sequences = [item[0] for item in batch]  # List of graph sequences
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    video_ids = [item[2] for item in batch]
    
    # Get sequence length (should be same for all)
    seq_len = len(sequences[0])
    
    # Batch graphs at each timestep
    batched_sequences = []
    for t in range(seq_len):
        # Get all graphs at timestep t
        graphs_at_t = [seq[t] for seq in sequences]
        # Batch them together
        batched_graph = Batch.from_data_list(graphs_at_t)
        batched_sequences.append(batched_graph)
    
    return batched_sequences, labels, video_ids


class TemporalExerciseTrainer:
    """Training pipeline for temporal model"""
    
    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def prepare_data(self, window_size=30, stride=15):
        """Load or create dataset"""
        print("=== Temporal Exercise Recognition Training ===\n")
        
        csv_path = self.dataset_path / "processed" / "pose_features.csv"
        
        if not csv_path.exists():
            print("Step 1: Extracting pose features with MediaPipe...")
            extractor = MediaPipeFeatureExtractor()
            df, exercise_to_label = extractor.process_dataset(self.dataset_path)
        else:
            print("Step 1: Loading existing pose features...")
            df = pd.read_csv(csv_path)
            exercise_names = df['class'].unique()
            exercise_to_label = {name: i for i, name in enumerate(exercise_names)}
        
        print(f"Features loaded: {len(df)} frames")
        print(f"Exercises: {list(exercise_to_label.keys())}")
        
        # Create temporal window dataset
        print(f"\nStep 2: Creating temporal window dataset...")
        dataset = TemporalWindowDataset(csv_path, exercise_to_label, window_size, stride)
        
        return dataset, exercise_to_label
    
    def split_dataset(self, dataset, test_size=0.2):
        """Split into train and test"""
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        
        train_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders with custom collate
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=16,  # Smaller batch size for temporal data
            shuffle=True,
            collate_fn=collate_temporal_batch
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=16, 
            shuffle=False,
            collate_fn=collate_temporal_batch
        )
        
        print(f"Train windows: {len(train_dataset)}")
        print(f"Test windows: {len(test_dataset)}")
        
        return train_loader, test_loader
    
    def train(self, train_loader, test_loader, exercise_to_label, epochs=100):
        """Train the temporal model"""
        print(f"\nStep 3: Initializing Temporal GNN+LSTM model...")
        
        model = TemporalExerciseModel(
            num_features=7,
            gnn_hidden=128,
            gnn_output=64,
            lstm_hidden=128,
            lstm_layers=2,
            num_classes=len(exercise_to_label)
        ).to(self.device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Training on: {self.device}")
        
        # Training loop
        print("\nStep 4: Training model...")
        train_losses = []
        train_accuracies = []
        best_accuracy = 0.0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
            for graph_sequences, labels, video_ids in progress_bar:
                # Move graphs to device
                graph_sequences = [g.to(self.device) for g in graph_sequences]
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                output = model(graph_sequences)
                loss = criterion(output, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100. * correct / total:.2f}%'
                })
            
            avg_loss = total_loss / len(train_loader)
            accuracy = correct / total
            
            train_losses.append(avg_loss)
            train_accuracies.append(accuracy)
            
            print(f'Epoch {epoch+1}/{epochs}: Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
            
            # Save best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'exercise_to_label': exercise_to_label,
                    'epoch': epoch,
                    'accuracy': accuracy
                }, self.dataset_path / "temporal_model_best.pth")
        
        # Evaluation
        print("\nStep 5: Evaluating model...")
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for graph_sequences, labels, video_ids in tqdm(test_loader, desc='Evaluating'):
                graph_sequences = [g.to(self.device) for g in graph_sequences]
                labels = labels.to(self.device)
                
                output = model(graph_sequences)
                _, predicted = torch.max(output, 1)
                
                y_true.extend(labels.cpu().numpy())
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
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=exercise_names, yticklabels=exercise_names)
        plt.title('Confusion Matrix - Temporal Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'confusion_matrix_temporal.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot training curves
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies)
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.dataset_path / 'training_curves_temporal.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\nModel saved to {self.dataset_path / 'temporal_model_best.pth'}")
        print(f"Training complete! Achieved {accuracy*100:.2f}% accuracy")
        
        return model, exercise_to_label, accuracy


if __name__ == "__main__":
    print("=== Temporal Exercise Recognition with GNN + LSTM ===")
    print("Window-based prediction for real-time applications")
    
    # Set your dataset path
    dataset_path = r"C:\Users\User\.cache\kagglehub\datasets\data\exercise-videos\real-time-exercise-recognition-dataset\versions\1.0.0\training-data"
    
    # Initialize trainer
    trainer = TemporalExerciseTrainer(dataset_path)
    
    # Prepare data with 30-frame windows (1 second at 30fps)
    dataset, exercise_to_label = trainer.prepare_data(window_size=30, stride=15)
    
    # Split data
    train_loader, test_loader = trainer.split_dataset(dataset)
    
    # Train model
    model, exercise_mapping, accuracy = trainer.train(
        train_loader, 
        test_loader, 
        exercise_to_label, 
        epochs=50
    )
    
    print(f"\nTraining completed!")
    print(f"Final accuracy: {accuracy*100:.2f}%")
    print(f"Exercise mapping: {exercise_mapping}")
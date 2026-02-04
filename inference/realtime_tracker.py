import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
import json
from typing import Optional, Dict
import torch
from rep_counter import RepCounter, ExerciseType, MultiExerciseTracker
from train import ExerciseGNN, VideoGraphDataset


class RealTimeExerciseTracker:
    """
    Real-time exercise detection and rep counting system.
    Combines pose detection, exercise classification, and rep counting.
    """
    
    def __init__(
        self, 
        model_path: str,
        exercise_mapping: Dict[str, int],
        confidence_threshold: float = 0.7,
        fps: int = 30
    ):
        """
        Initialize real-time tracker.
        
        Args:
            model_path: Path to trained exercise classification model
            exercise_mapping: Dictionary mapping exercise names to label IDs
            confidence_threshold: Minimum confidence for exercise classification
            fps: Frames per second for video processing
        """
        self.fps = fps
        self.confidence_threshold = confidence_threshold
        
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Load exercise classification model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path, len(exercise_mapping))
        self.exercise_mapping = exercise_mapping
        self.reverse_mapping = {v: k for k, v in exercise_mapping.items()}
        
        # Initialize multi-exercise tracker
        self.tracker = MultiExerciseTracker(fps=fps)
        
        # Classification smoothing
        self.classification_history = []
        self.classification_window = 10  # Smooth over 10 frames
        
        # Performance tracking
        self.frame_times = []
        
    def _load_model(self, model_path: str, num_classes: int) -> torch.nn.Module:
        """Load the trained exercise classification model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = ExerciseGNN(
            num_features=7,  # x, y, z + 4 orientation features
            hidden_dim=128,
            num_classes=num_classes
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        return model
    
    def _landmarks_to_array(self, landmarks) -> np.ndarray:
        """Convert MediaPipe landmarks to numpy array [33, 3]"""
        coords = np.zeros((33, 3))
        for i, landmark in enumerate(landmarks.landmark):
            coords[i] = [landmark.x, landmark.y, landmark.z]
        return coords
    
    def _classify_exercise(self, landmarks: np.ndarray) -> tuple[Optional[str], float]:
        """
        Classify the exercise being performed.
        
        Args:
            landmarks: [33, 3] pose keypoints
            
        Returns:
            (exercise_name, confidence) or (None, 0.0)
        """
        try:
            # Create simple graph representation for single frame
            # Average would normally be over multiple frames, but for real-time
            # we use current frame with simplified approach
            node_features = landmarks  # [33, 3]
            
            # Add dummy orientation features for consistency with training
            orientation = np.zeros((33, 4))  # Placeholder
            node_features = np.concatenate([node_features, orientation], axis=1)  # [33, 7]
            
            # Create edge connections (same as in training)
            edge_connections = [
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                (9, 10), (11, 12), (11, 23), (12, 24), (23, 24),
                (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
                (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
                (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
                (24, 26), (26, 28), (28, 30), (28, 32), (30, 32)
            ]
            
            edges = []
            for src, dst in edge_connections:
                if src < 33 and dst < 33:
                    edges.append([src, dst])
                    edges.append([dst, src])
            
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Create PyG Data object
            x = torch.tensor(node_features, dtype=torch.float32).to(self.device)
            edge_index = edge_index.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(x, edge_index, batch=None)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_label = predicted.item()
            
            # Add to classification history for smoothing
            self.classification_history.append(predicted_label)
            if len(self.classification_history) > self.classification_window:
                self.classification_history.pop(0)
            
            # Use most common classification in recent history
            if len(self.classification_history) >= 3:
                most_common = max(set(self.classification_history), 
                                key=self.classification_history.count)
                exercise_name = self.reverse_mapping[most_common]
            else:
                exercise_name = self.reverse_mapping[predicted_label]
            
            return exercise_name, confidence
            
        except Exception as e:
            print(f"Classification error: {e}")
            return None, 0.0
    
    def _map_exercise_to_type(self, exercise_name: str) -> Optional[ExerciseType]:
        """Map exercise name to ExerciseType enum"""
        mapping = {
            'squat': ExerciseType.SQUAT,
            'push-up': ExerciseType.PUSH_UP,
            'barbell biceps curl': ExerciseType.BICEP_CURL,
            'hammer curl': ExerciseType.HAMMER_CURL,
            'shoulder press': ExerciseType.SHOULDER_PRESS,
        }
        return mapping.get(exercise_name.lower())
    
    def process_frame(self, frame: np.ndarray, timestamp: float) -> Dict:
        """
        Process a single frame for exercise tracking.
        
        Args:
            frame: BGR image from camera
            timestamp: Current timestamp in seconds
            
        Returns:
            Dictionary with tracking results and annotated frame
        """
        start_time = time.time()
        
        # Convert to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect pose
        results = self.pose.process(rgb_frame)
        
        tracking_data = {
            'pose_detected': False,
            'exercise': None,
            'confidence': 0.0,
            'rep_count': 0,
            'state': 'waiting',
            'form_score': 0.0,
            'frame': frame
        }
        
        if results.pose_landmarks:
            tracking_data['pose_detected'] = True
            
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Convert landmarks to array
            landmarks = self._landmarks_to_array(results.pose_landmarks)
            
            # Classify exercise
            exercise_name, confidence = self._classify_exercise(landmarks)
            
            if exercise_name and confidence >= self.confidence_threshold:
                tracking_data['exercise'] = exercise_name
                tracking_data['confidence'] = confidence
                
                # Map to ExerciseType
                exercise_type = self._map_exercise_to_type(exercise_name)
                
                if exercise_type:
                    # Update rep counter
                    status = self.tracker.update(landmarks, exercise_type, timestamp)
                    
                    tracking_data['rep_count'] = status['rep_count']
                    tracking_data['state'] = status['state']
                    
                    # Get latest rep metrics if available
                    if self.tracker.current_counter and self.tracker.current_counter.completed_reps:
                        latest_rep = self.tracker.current_counter.completed_reps[-1]
                        tracking_data['form_score'] = latest_rep.form_score
                        tracking_data['last_rep_rom'] = latest_rep.range_of_motion
                        tracking_data['last_rep_duration'] = latest_rep.duration
            
            # Draw tracking info on frame
            self._draw_tracking_info(frame, tracking_data)
        
        # Calculate FPS
        frame_time = time.time() - start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        avg_fps = 1.0 / (sum(self.frame_times) / len(self.frame_times))
        tracking_data['fps'] = avg_fps
        
        return tracking_data
    
    def _draw_tracking_info(self, frame: np.ndarray, data: Dict):
        """Draw tracking information overlay on frame"""
        h, w = frame.shape[:2]
        
        # Background for text
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 200), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        y_offset = 40
        line_height = 35
        
        # Exercise name
        if data['exercise']:
            text = f"Exercise: {data['exercise']}"
            cv2.putText(frame, text, (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += line_height
            
            # Confidence
            text = f"Confidence: {data['confidence']:.2%}"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += line_height
        else:
            text = "No exercise detected"
            cv2.putText(frame, text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            y_offset += line_height
        
        # Rep count (large and prominent)
        if data['rep_count'] > 0:
            rep_text = f"Reps: {data['rep_count']}"
            cv2.putText(frame, rep_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            y_offset += line_height + 10
        
        # State
        state_colors = {
            'rest': (128, 128, 128),
            'top': (0, 255, 0),
            'descending': (0, 165, 255),
            'bottom': (0, 0, 255),
            'ascending': (255, 165, 0)
        }
        state_color = state_colors.get(data['state'], (255, 255, 255))
        cv2.putText(frame, f"State: {data['state']}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)
        y_offset += line_height
        
        # Form score
        if data['form_score'] > 0:
            form_color = self._get_form_color(data['form_score'])
            cv2.putText(frame, f"Form: {data['form_score']:.1f}/100", (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, form_color, 2)
        
        # FPS counter (top right)
        if 'fps' in data:
            cv2.putText(frame, f"FPS: {data['fps']:.1f}", (w - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def _get_form_color(self, score: float) -> tuple:
        """Get color based on form score"""
        if score >= 85:
            return (0, 255, 0)  # Green - excellent
        elif score >= 70:
            return (0, 255, 255)  # Yellow - good
        elif score >= 50:
            return (0, 165, 255)  # Orange - needs improvement
        else:
            return (0, 0, 255)  # Red - poor form
    
    def run_camera(self, camera_id: int = 0, save_video: bool = False):
        """
        Run real-time tracking from camera feed.
        
        Args:
            camera_id: Camera device ID
            save_video: Whether to save the annotated video
        """
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('workout_tracking.mp4', fourcc, self.fps,
                                (int(cap.get(3)), int(cap.get(4))))
        
        print("Starting real-time exercise tracking...")
        print("Press 'q' to quit")
        print("Press 'r' to reset rep counter")
        print("Press 's' to save current set metrics")
        
        start_time = time.time()
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                current_time = time.time() - start_time
                
                # Process frame
                results = self.process_frame(frame, current_time)
                annotated_frame = results['frame']
                
                # Display
                cv2.imshow('Exercise Tracking', annotated_frame)
                
                if save_video:
                    out.write(annotated_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    if self.tracker.current_counter:
                        self.tracker.current_counter.reset()
                        print("Rep counter reset")
                elif key == ord('s'):
                    self._save_current_metrics()
        
        finally:
            cap.release()
            if save_video:
                out.release()
            cv2.destroyAllWindows()
            
            # Print final summary
            self._print_session_summary()
    
    def run_video(self, video_path: str, save_output: bool = True):
        """
        Run tracking on a video file.
        
        Args:
            video_path: Path to input video
            save_output: Whether to save annotated output
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        
        output_path = None
        if save_output:
            output_path = Path(video_path).stem + '_tracked.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                (int(cap.get(3)), int(cap.get(4))))
        
        print(f"Processing video: {video_path}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_count = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            timestamp = frame_count / fps
            results = self.process_frame(frame, timestamp)
            
            if save_output:
                out.write(results['frame'])
            
            frame_count += 1
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% - Reps: {results['rep_count']}")
        
        processing_time = time.time() - start_time
        
        cap.release()
        if save_output:
            out.release()
        
        print(f"\nProcessing complete!")
        print(f"Processed {frame_count} frames in {processing_time:.2f}s")
        print(f"Average FPS: {frame_count / processing_time:.2f}")
        if output_path:
            print(f"Output saved to: {output_path}")
        
        self._print_session_summary()
    
    def _save_current_metrics(self):
        """Save current set metrics to JSON"""
        if self.tracker.current_counter:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"set_metrics_{timestamp}.json"
            self.tracker.current_counter.export_metrics(filename)
            print(f"Metrics saved to {filename}")
    
    def _print_session_summary(self):
        """Print summary of tracking session"""
        summary = self.tracker.get_workout_summary()
        
        print("\n" + "="*50)
        print("WORKOUT SESSION SUMMARY")
        print("="*50)
        
        if summary['total_exercises'] == 0:
            print("No exercises completed")
            return
        
        for exercise_data in summary['exercises']:
            print(f"\nExercise: {exercise_data['exercise']}")
            print(f"  Total Reps: {exercise_data['total_reps']}")
            print(f"  Avg Form Score: {exercise_data['average_form_score']:.1f}/100")
            print(f"  Duration: {exercise_data['total_duration']:.1f}s")
        
        # Save full workout summary
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        summary_file = f"workout_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nFull summary saved to: {summary_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time exercise tracking')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--mode', type=str, choices=['camera', 'video'],
                       default='camera', help='Tracking mode')
    parser.add_argument('--video', type=str, help='Video file path (for video mode)')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--save', action='store_true', help='Save output video')
    
    args = parser.parse_args()
    
    # Exercise mapping (should match your training)
    exercise_mapping = {
        "barbell biceps curl": 0,
        "hammer curl": 1,
        "push-up": 2,
        "shoulder press": 3,
        "squat": 4
    }
    
    # Initialize tracker
    tracker = RealTimeExerciseTracker(
        model_path=args.model,
        exercise_mapping=exercise_mapping,
        confidence_threshold=0.6,
        fps=30
    )
    
    # Run tracking
    if args.mode == 'camera':
        tracker.run_camera(camera_id=args.camera, save_video=args.save)
    else:
        if not args.video:
            print("Error: --video required for video mode")
        else:
            tracker.run_video(args.video, save_output=args.save)
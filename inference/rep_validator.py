import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from typing import List, Dict, Tuple
import mediapipe as mp
from rep_counter import RepCounter, ExerciseType, RepMetrics


class RepCounterValidator:
    """
    Validate rep counting accuracy against ground truth annotations.
    Useful for testing and tuning the algorithm.
    """
    
    def __init__(self, video_path: str, ground_truth_reps: int, exercise_type: ExerciseType):
        self.video_path = video_path
        self.ground_truth_reps = ground_truth_reps
        self.exercise_type = exercise_type
        
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize rep counter
        self.counter = RepCounter(exercise_type, fps=30)
        
    def extract_landmarks_from_video(self) -> List[np.ndarray]:
        """Extract pose landmarks from entire video"""
        cap = cv2.VideoCapture(self.video_path)
        landmarks_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                coords = np.zeros((33, 3))
                for i, landmark in enumerate(results.pose_landmarks.landmark):
                    coords[i] = [landmark.x, landmark.y, landmark.z]
                landmarks_sequence.append(coords)
        
        cap.release()
        return landmarks_sequence
    
    def validate(self) -> Dict:
        """Run validation and return metrics"""
        print(f"Validating: {self.video_path}")
        print(f"Ground truth: {self.ground_truth_reps} reps")
        print(f"Exercise: {self.exercise_type.value}")
        
        # Extract landmarks
        landmarks_sequence = self.extract_landmarks_from_video()
        print(f"Extracted {len(landmarks_sequence)} frames")
        
        # Process through rep counter
        for frame_idx, landmarks in enumerate(landmarks_sequence):
            timestamp = frame_idx / 30.0  # Assuming 30fps
            self.counter.update(landmarks, timestamp)
        
        # Get results
        detected_reps = self.counter.rep_count
        set_metrics = self.counter.get_set_summary()
        
        # Calculate accuracy
        accuracy = min(detected_reps, self.ground_truth_reps) / max(detected_reps, self.ground_truth_reps)
        error = abs(detected_reps - self.ground_truth_reps)
        
        results = {
            'video': str(self.video_path),
            'exercise': self.exercise_type.value,
            'ground_truth': self.ground_truth_reps,
            'detected': detected_reps,
            'error': error,
            'accuracy': accuracy,
            'avg_form_score': set_metrics.average_form_score,
            'avg_rom': set_metrics.average_rom,
            'consistency': set_metrics.consistency_score
        }
        
        print(f"\nResults:")
        print(f"  Detected: {detected_reps} reps")
        print(f"  Error: {error} reps")
        print(f"  Accuracy: {accuracy:.2%}")
        print(f"  Avg Form Score: {set_metrics.average_form_score:.1f}/100")
        
        return results


class RepCounterVisualizer:
    """
    Visualize rep counting metrics and performance.
    """
    
    def __init__(self, output_dir: str = "visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
    
    def plot_rep_metrics(self, rep_details: List[RepMetrics], title: str = "Rep Analysis"):
        """Plot detailed metrics for each rep in a set"""
        if not rep_details:
            print("No rep data to visualize")
            return
        
        # Extract data
        rep_numbers = [rep.rep_number for rep in rep_details]
        durations = [rep.duration for rep in rep_details]
        form_scores = [rep.form_score for rep in rep_details]
        roms = [rep.range_of_motion for rep in rep_details]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # 1. Duration per rep
        axes[0, 0].bar(rep_numbers, durations, color='steelblue', alpha=0.7)
        axes[0, 0].axhline(np.mean(durations), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(durations):.2f}s')
        axes[0, 0].set_xlabel('Rep Number')
        axes[0, 0].set_ylabel('Duration (seconds)')
        axes[0, 0].set_title('Rep Duration')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Form score per rep
        colors = ['green' if score >= 85 else 'gold' if score >= 70 else 'orange' 
                 if score >= 50 else 'red' for score in form_scores]
        axes[0, 1].bar(rep_numbers, form_scores, color=colors, alpha=0.7)
        axes[0, 1].axhline(85, color='green', linestyle='--', label='Excellent (85+)')
        axes[0, 1].axhline(70, color='gold', linestyle='--', label='Good (70+)')
        axes[0, 1].set_xlabel('Rep Number')
        axes[0, 1].set_ylabel('Form Score')
        axes[0, 1].set_title('Form Quality')
        axes[0, 1].set_ylim([0, 100])
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Range of motion
        axes[1, 0].plot(rep_numbers, roms, marker='o', linewidth=2, 
                       markersize=8, color='purple')
        axes[1, 0].fill_between(rep_numbers, roms, alpha=0.3, color='purple')
        axes[1, 0].axhline(np.mean(roms), color='red', linestyle='--',
                          label=f'Mean: {np.mean(roms):.1f}°')
        axes[1, 0].set_xlabel('Rep Number')
        axes[1, 0].set_ylabel('Range of Motion (degrees)')
        axes[1, 0].set_title('Range of Motion Consistency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Tempo breakdown
        eccentric_times = [rep.tempo['eccentric'] for rep in rep_details]
        concentric_times = [rep.tempo['concentric'] for rep in rep_details]
        
        x = np.arange(len(rep_numbers))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, eccentric_times, width, label='Eccentric (lowering)',
                      color='coral', alpha=0.7)
        axes[1, 1].bar(x + width/2, concentric_times, width, label='Concentric (lifting)',
                      color='skyblue', alpha=0.7)
        axes[1, 1].set_xlabel('Rep Number')
        axes[1, 1].set_ylabel('Time (seconds)')
        axes[1, 1].set_title('Tempo Analysis')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(rep_numbers)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        output_path = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
        
        plt.show()
    
    def plot_angle_trajectory(self, angles: List[float], states: List[str], 
                             fps: int = 30, title: str = "Joint Angle Trajectory"):
        """Plot joint angle over time with state annotations"""
        if not angles:
            print("No angle data to visualize")
            return
        
        time_points = [i / fps for i in range(len(angles))]
        
        fig, ax = plt.subplots(figsize=(15, 6))
        
        # Plot angle trajectory
        ax.plot(time_points, angles, linewidth=2, color='steelblue', label='Joint Angle')
        
        # Color background by state
        state_colors = {
            'rest': 'gray',
            'top': 'lightgreen',
            'descending': 'lightcoral',
            'bottom': 'lightblue',
            'ascending': 'lightyellow'
        }
        
        current_state = states[0] if states else 'rest'
        start_idx = 0
        
        for i, state in enumerate(states):
            if state != current_state or i == len(states) - 1:
                # Fill region with state color
                end_idx = i if i == len(states) - 1 else i - 1
                ax.axvspan(time_points[start_idx], time_points[end_idx],
                          alpha=0.3, color=state_colors.get(current_state, 'white'),
                          label=current_state if start_idx == 0 else "")
                current_state = state
                start_idx = i
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Joint Angle (degrees)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        
        output_path = self.output_dir / f"{title.replace(' ', '_').lower()}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved trajectory to: {output_path}")
        
        plt.show()
    
    def plot_workout_summary(self, workout_data: Dict):
        """Visualize entire workout session"""
        if not workout_data.get('exercises'):
            print("No workout data to visualize")
            return
        
        exercises = workout_data['exercises']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Workout Summary', fontsize=16, fontweight='bold')
        
        # Extract data
        exercise_names = [ex['exercise'] for ex in exercises]
        total_reps = [ex['total_reps'] for ex in exercises]
        form_scores = [ex['average_form_score'] for ex in exercises]
        durations = [ex['total_duration'] for ex in exercises]
        
        # 1. Total reps by exercise
        axes[0].bar(exercise_names, total_reps, color='steelblue', alpha=0.7)
        axes[0].set_xlabel('Exercise')
        axes[0].set_ylabel('Total Reps')
        axes[0].set_title('Reps per Exercise')
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Average form scores
        colors = ['green' if score >= 85 else 'gold' if score >= 70 else 'orange'
                 for score in form_scores]
        axes[1].bar(exercise_names, form_scores, color=colors, alpha=0.7)
        axes[1].axhline(85, color='green', linestyle='--', linewidth=1)
        axes[1].axhline(70, color='gold', linestyle='--', linewidth=1)
        axes[1].set_xlabel('Exercise')
        axes[1].set_ylabel('Form Score')
        axes[1].set_title('Average Form Quality')
        axes[1].set_ylim([0, 100])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. Duration by exercise
        axes[2].bar(exercise_names, durations, color='coral', alpha=0.7)
        axes[2].set_xlabel('Exercise')
        axes[2].set_ylabel('Duration (seconds)')
        axes[2].set_title('Time per Exercise')
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "workout_summary.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved workout summary to: {output_path}")
        
        plt.show()
    
    def create_comparison_report(self, validation_results: List[Dict]):
        """Create comparison report across multiple videos"""
        if not validation_results:
            print("No validation results to compare")
            return
        
        df = pd.DataFrame(validation_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Rep Counter Validation Report', fontsize=16, fontweight='bold')
        
        # 1. Accuracy distribution
        axes[0, 0].hist(df['accuracy'] * 100, bins=10, color='steelblue', 
                       alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(df['accuracy'].mean() * 100, color='red', 
                          linestyle='--', linewidth=2, 
                          label=f"Mean: {df['accuracy'].mean()*100:.1f}%")
        axes[0, 0].set_xlabel('Accuracy (%)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Accuracy Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3, axis='y')
        
        # 2. Error analysis
        axes[0, 1].scatter(df['ground_truth'], df['detected'], 
                          alpha=0.6, s=100, color='steelblue')
        max_reps = max(df['ground_truth'].max(), df['detected'].max())
        axes[0, 1].plot([0, max_reps], [0, max_reps], 'r--', linewidth=2, 
                       label='Perfect Accuracy')
        axes[0, 1].set_xlabel('Ground Truth Reps')
        axes[0, 1].set_ylabel('Detected Reps')
        axes[0, 1].set_title('Detection Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error by exercise type
        exercise_errors = df.groupby('exercise')['error'].mean()
        axes[1, 0].bar(exercise_errors.index, exercise_errors.values, 
                      color='coral', alpha=0.7)
        axes[1, 0].set_xlabel('Exercise Type')
        axes[1, 0].set_ylabel('Average Error (reps)')
        axes[1, 0].set_title('Average Error by Exercise')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        # 4. Form score distribution
        axes[1, 1].boxplot([df[df['exercise'] == ex]['avg_form_score'].values 
                           for ex in df['exercise'].unique()],
                          labels=df['exercise'].unique())
        axes[1, 1].set_xlabel('Exercise Type')
        axes[1, 1].set_ylabel('Form Score')
        axes[1, 1].set_title('Form Score Distribution')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = self.output_dir / "validation_report.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved validation report to: {output_path}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("VALIDATION SUMMARY")
        print("="*50)
        print(f"Total videos tested: {len(df)}")
        print(f"Average accuracy: {df['accuracy'].mean()*100:.2f}%")
        print(f"Average error: {df['error'].mean():.2f} reps")
        print(f"Perfect accuracy rate: {(df['error'] == 0).sum() / len(df) * 100:.1f}%")
        print(f"Within ±1 rep: {(df['error'] <= 1).sum() / len(df) * 100:.1f}%")
        print(f"Average form score: {df['avg_form_score'].mean():.1f}/100")
        
        plt.show()


def run_validation_suite(video_dir: str, ground_truth_file: str):
    """
    Run validation on a suite of test videos.
    
    Args:
        video_dir: Directory containing test videos
        ground_truth_file: JSON file with ground truth annotations
            Format: {"video_name.mp4": {"reps": 10, "exercise": "squat"}, ...}
    """
    # Load ground truth
    with open(ground_truth_file, 'r') as f:
        ground_truth = json.load(f)
    
    video_dir = Path(video_dir)
    results = []
    
    # Run validation on each video
    for video_file, annotations in ground_truth.items():
        video_path = video_dir / video_file
        
        if not video_path.exists():
            print(f"Warning: {video_path} not found, skipping")
            continue
        
        # Map exercise name to ExerciseType
        exercise_map = {
            'squat': ExerciseType.SQUAT,
            'push-up': ExerciseType.PUSH_UP,
            'bicep_curl': ExerciseType.BICEP_CURL,
            'hammer_curl': ExerciseType.HAMMER_CURL,
            'shoulder_press': ExerciseType.SHOULDER_PRESS,
        }
        
        exercise_type = exercise_map.get(annotations['exercise'].lower())
        if not exercise_type:
            print(f"Warning: Unknown exercise type {annotations['exercise']}, skipping")
            continue
        
        # Run validation
        validator = RepCounterValidator(
            str(video_path),
            annotations['reps'],
            exercise_type
        )
        
        result = validator.validate()
        results.append(result)
        print("\n" + "-"*50 + "\n")
    
    # Visualize results
    visualizer = RepCounterVisualizer()
    visualizer.create_comparison_report(results)
    
    # Save results to JSON
    with open('validation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


if __name__ == "__main__":
    # Example: Visualize rep metrics from a saved JSON file
    visualizer = RepCounterVisualizer()
    
    print("Rep Counter Visualization Tool")
    print("1. Visualize set metrics from JSON")
    print("2. Run validation suite")
    print("3. Exit")
    
    choice = input("Enter choice: ")
    
    if choice == "1":
        json_path = input("Enter path to set metrics JSON: ")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert to RepMetrics objects
        rep_details = []
        for rep in data.get('rep_details', []):
            rep_metrics = RepMetrics(
                rep_number=rep['rep_number'],
                duration=rep['duration'],
                range_of_motion=rep['range_of_motion'],
                form_score=rep['form_score'],
                tempo=rep['tempo'],
                peak_angle=rep['peak_angle'],
                bottom_angle=rep['bottom_angle'],
                timestamp=rep['timestamp']
            )
            rep_details.append(rep_metrics)
        
        visualizer.plot_rep_metrics(rep_details, 
                                   f"{data['exercise_type']} Set Analysis")
    
    elif choice == "2":
        video_dir = input("Enter video directory: ")
        ground_truth = input("Enter ground truth JSON file: ")
        run_validation_suite(video_dir, ground_truth)
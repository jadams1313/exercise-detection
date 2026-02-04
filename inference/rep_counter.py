import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from enum import Enum
import json


class ExerciseType(Enum):
    """Supported exercise types with their specific tracking requirements"""
    SQUAT = "squat"
    PUSH_UP = "push_up"
    BICEP_CURL = "bicep_curl"
    HAMMER_CURL = "hammer_curl"
    SHOULDER_PRESS = "shoulder_press"
    DEADLIFT = "deadlift"
    BENCH_PRESS = "bench_press"
    PULL_UP = "pull_up"


@dataclass
class RepMetrics:
    """Metrics for a single repetition"""
    rep_number: int
    duration: float  # seconds
    range_of_motion: float  # degrees or normalized distance
    form_score: float  # 0-100
    tempo: Dict[str, float]  # eccentric, pause, concentric phases
    peak_angle: float
    bottom_angle: float
    timestamp: float


@dataclass
class SetMetrics:
    """Aggregated metrics for a complete set"""
    total_reps: int
    valid_reps: int
    invalid_reps: int
    average_rom: float
    average_form_score: float
    average_tempo: float
    consistency_score: float  # How consistent were the reps
    total_duration: float
    rep_details: List[RepMetrics]


class RepCounter:
    """
    Advanced rep counting algorithm using biomechanical joint angles and movement patterns.
    
    Uses a state machine approach with hysteresis to avoid false counting and 
    tracks multiple metrics per rep for comprehensive analytics.
    """
    
    def __init__(self, exercise_type: ExerciseType, fps: int = 30):
        self.exercise_type = exercise_type
        self.fps = fps
        
        # Rep counting state
        self.rep_count = 0
        self.current_state = "rest"  # rest, descending, bottom, ascending, top
        self.frame_count = 0
        
        # Movement tracking
        self.angle_history = deque(maxlen=fps * 2)  # 2 seconds of history
        self.state_start_frame = 0
        self.rep_start_frame = 0
        
        # Current rep tracking
        self.current_rep_angles = []
        self.current_rep_positions = []
        self.peak_angle = None
        self.bottom_angle = None
        
        # Completed reps storage
        self.completed_reps: List[RepMetrics] = []
        
        # Exercise-specific thresholds
        self.thresholds = self._get_exercise_thresholds()
        
        # Smoothing and noise reduction
        self.angle_buffer = deque(maxlen=5)  # For smoothing
        
    def _get_exercise_thresholds(self) -> Dict:
        """Get exercise-specific thresholds for rep counting"""
        thresholds = {
            ExerciseType.SQUAT: {
                'start_angle': 170,  # Nearly straight legs
                'bottom_angle': 90,  # Parallel or below
                'hysteresis': 15,  # Degrees of hysteresis
                'min_rom': 60,  # Minimum range of motion
                'max_duration': 5.0,  # Max seconds per rep
                'min_duration': 0.5,  # Min seconds per rep
            },
            ExerciseType.PUSH_UP: {
                'start_angle': 160,  # Arms nearly straight
                'bottom_angle': 90,  # Elbow at 90 degrees
                'hysteresis': 20,
                'min_rom': 50,
                'max_duration': 4.0,
                'min_duration': 0.4,
            },
            ExerciseType.BICEP_CURL: {
                'start_angle': 160,  # Arm extended
                'bottom_angle': 50,  # Fully curled
                'hysteresis': 15,
                'min_rom': 90,
                'max_duration': 4.0,
                'min_duration': 0.5,
            },
            ExerciseType.HAMMER_CURL: {
                'start_angle': 160,
                'bottom_angle': 50,
                'hysteresis': 15,
                'min_rom': 90,
                'max_duration': 4.0,
                'min_duration': 0.5,
            },
            ExerciseType.SHOULDER_PRESS: {
                'start_angle': 90,  # Arms at 90 degrees (starting position)
                'bottom_angle': 160,  # Arms extended overhead
                'hysteresis': 15,
                'min_rom': 60,
                'max_duration': 4.0,
                'min_duration': 0.5,
            },
            ExerciseType.DEADLIFT: {
                'start_angle': 170,  # Nearly straight back
                'bottom_angle': 70,  # Bent at hips
                'hysteresis': 20,
                'min_rom': 70,
                'max_duration': 5.0,
                'min_duration': 0.8,
            },
        }
        
        return thresholds.get(self.exercise_type, thresholds[ExerciseType.SQUAT])
    
    def calculate_angle(self, point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
        """
        Calculate angle between three points (in degrees).
        point2 is the vertex of the angle.
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Calculate angle using dot product
        cos_angle = np.dot(vector1, vector2) / (
            np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-8
        )
        
        # Clamp to valid range for arccos
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return np.degrees(angle)
    
    def extract_exercise_angle(self, landmarks: np.ndarray) -> Optional[float]:
        """
        Extract the primary tracking angle for the exercise.
        landmarks: [33, 3] array of pose keypoints (x, y, z)
        
        Returns angle in degrees or None if landmarks are invalid
        """
        try:
            if self.exercise_type == ExerciseType.SQUAT:
                # Hip-Knee-Ankle angle (right leg)
                hip = landmarks[24]  # Right hip
                knee = landmarks[26]  # Right knee
                ankle = landmarks[28]  # Right ankle
                return self.calculate_angle(hip, knee, ankle)
            
            elif self.exercise_type == ExerciseType.PUSH_UP:
                # Shoulder-Elbow-Wrist angle (right arm)
                shoulder = landmarks[12]  # Right shoulder
                elbow = landmarks[14]  # Right elbow
                wrist = landmarks[16]  # Right wrist
                return self.calculate_angle(shoulder, elbow, wrist)
            
            elif self.exercise_type in [ExerciseType.BICEP_CURL, ExerciseType.HAMMER_CURL]:
                # Shoulder-Elbow-Wrist angle (right arm)
                shoulder = landmarks[12]
                elbow = landmarks[14]
                wrist = landmarks[16]
                return self.calculate_angle(shoulder, elbow, wrist)
            
            elif self.exercise_type == ExerciseType.SHOULDER_PRESS:
                # Elbow-Shoulder-Hip angle (right side)
                elbow = landmarks[14]
                shoulder = landmarks[12]
                hip = landmarks[24]
                return self.calculate_angle(elbow, shoulder, hip)
            
            elif self.exercise_type == ExerciseType.DEADLIFT:
                # Hip-Shoulder-Neck angle (torso angle)
                hip = landmarks[24]
                shoulder = landmarks[12]
                # Use nose as proxy for head position
                nose = landmarks[0]
                return self.calculate_angle(hip, shoulder, nose)
            
            else:
                return None
                
        except Exception as e:
            return None
    
    def smooth_angle(self, angle: float) -> float:
        """Apply moving average smoothing to reduce noise"""
        self.angle_buffer.append(angle)
        return np.mean(list(self.angle_buffer))
    
    def calculate_form_score(self, rep_angles: List[float], duration: float) -> float:
        """
        Calculate form score (0-100) based on:
        - Range of motion
        - Movement smoothness
        - Tempo consistency
        - Depth achieved
        """
        if len(rep_angles) < 5:
            return 0.0
        
        score = 100.0
        thresholds = self.thresholds
        
        # 1. Range of motion score (40 points)
        rom = max(rep_angles) - min(rep_angles)
        rom_score = min(40, (rom / thresholds['min_rom']) * 40)
        
        # 2. Depth score (30 points) - did they go deep enough?
        min_angle = min(rep_angles)
        if min_angle <= thresholds['bottom_angle']:
            depth_score = 30.0
        else:
            # Penalize for not going deep enough
            depth_diff = min_angle - thresholds['bottom_angle']
            depth_score = max(0, 30 - (depth_diff / 2))
        
        # 3. Smoothness score (20 points) - penalize jerky movements
        angle_changes = np.diff(rep_angles)
        smoothness = np.std(angle_changes)
        smoothness_score = max(0, 20 - smoothness / 2)
        
        # 4. Tempo score (10 points) - reasonable duration
        if thresholds['min_duration'] <= duration <= thresholds['max_duration']:
            tempo_score = 10.0
        else:
            tempo_score = max(0, 10 - abs(duration - 2.0))
        
        total_score = rom_score + depth_score + smoothness_score + tempo_score
        return np.clip(total_score, 0, 100)
    
    def analyze_tempo(self, angles: List[float]) -> Dict[str, float]:
        """
        Analyze rep tempo: eccentric (lowering), pause, concentric (lifting).
        Returns duration of each phase in seconds.
        """
        if len(angles) < 5:
            return {'eccentric': 0, 'pause': 0, 'concentric': 0}
        
        # Find transition points
        max_idx = angles.index(max(angles))
        min_idx = angles.index(min(angles))
        
        # Ensure proper ordering
        if max_idx > min_idx:
            # Started high, went low, came back up
            eccentric_frames = min_idx - 0
            pause_frames = max_idx - min_idx if max_idx > min_idx else 0
            concentric_frames = len(angles) - max_idx
        else:
            # Started low, went up
            concentric_frames = max_idx - 0
            pause_frames = 0
            eccentric_frames = len(angles) - max_idx
        
        frame_duration = 1.0 / self.fps
        
        return {
            'eccentric': max(0, eccentric_frames * frame_duration),
            'pause': max(0, pause_frames * frame_duration),
            'concentric': max(0, concentric_frames * frame_duration)
        }
    
    def update(self, landmarks: np.ndarray, timestamp: float) -> Dict:
        """
        Main update function called for each frame.
        
        Args:
            landmarks: [33, 3] pose keypoints
            timestamp: current timestamp in seconds
            
        Returns:
            Dict with rep_count, state, and other metrics
        """
        self.frame_count += 1
        
        # Extract and smooth the tracking angle
        raw_angle = self.extract_exercise_angle(landmarks)
        if raw_angle is None:
            return self._get_status()
        
        angle = self.smooth_angle(raw_angle)
        self.angle_history.append(angle)
        
        # Store for current rep
        self.current_rep_angles.append(angle)
        self.current_rep_positions.append(landmarks.copy())
        
        # State machine for rep counting
        thresholds = self.thresholds
        current_duration = (self.frame_count - self.rep_start_frame) / self.fps
        
        # REST -> DESCENDING (exercise begins)
        if self.current_state == "rest":
            if angle >= thresholds['start_angle'] - thresholds['hysteresis']:
                self.current_state = "top"
                self.state_start_frame = self.frame_count
                self.rep_start_frame = self.frame_count
                self.current_rep_angles = [angle]
                self.current_rep_positions = [landmarks.copy()]
                self.peak_angle = angle
        
        # TOP -> DESCENDING (starting to lower)
        elif self.current_state == "top":
            if angle < thresholds['start_angle'] - thresholds['hysteresis']:
                self.current_state = "descending"
                self.state_start_frame = self.frame_count
                self.peak_angle = max(self.current_rep_angles)
        
        # DESCENDING -> BOTTOM (reached bottom position)
        elif self.current_state == "descending":
            if angle <= thresholds['bottom_angle'] + thresholds['hysteresis']:
                self.current_state = "bottom"
                self.state_start_frame = self.frame_count
                self.bottom_angle = angle
        
        # BOTTOM -> ASCENDING (starting to lift)
        elif self.current_state == "bottom":
            if angle > thresholds['bottom_angle'] + thresholds['hysteresis']:
                self.current_state = "ascending"
                self.state_start_frame = self.frame_count
                self.bottom_angle = min(self.current_rep_angles)
        
        # ASCENDING -> TOP (completed rep!)
        elif self.current_state == "ascending":
            if angle >= thresholds['start_angle'] - thresholds['hysteresis']:
                # Rep completed! Calculate metrics
                rep_duration = (self.frame_count - self.rep_start_frame) / self.fps
                
                # Validate rep
                rom = self.peak_angle - self.bottom_angle if self.peak_angle and self.bottom_angle else 0
                is_valid = (
                    rom >= thresholds['min_rom'] and
                    thresholds['min_duration'] <= rep_duration <= thresholds['max_duration']
                )
                
                if is_valid:
                    self.rep_count += 1
                    
                    # Calculate detailed metrics
                    form_score = self.calculate_form_score(
                        self.current_rep_angles, 
                        rep_duration
                    )
                    tempo = self.analyze_tempo(self.current_rep_angles)
                    
                    # Store rep metrics
                    rep_metrics = RepMetrics(
                        rep_number=self.rep_count,
                        duration=rep_duration,
                        range_of_motion=rom,
                        form_score=form_score,
                        tempo=tempo,
                        peak_angle=self.peak_angle,
                        bottom_angle=self.bottom_angle,
                        timestamp=timestamp
                    )
                    self.completed_reps.append(rep_metrics)
                
                # Reset for next rep
                self.current_state = "top"
                self.rep_start_frame = self.frame_count
                self.state_start_frame = self.frame_count
                self.current_rep_angles = [angle]
                self.current_rep_positions = [landmarks.copy()]
                self.peak_angle = angle
                self.bottom_angle = None
        
        # Timeout check - reset if taking too long
        if current_duration > thresholds['max_duration'] * 2:
            self.current_state = "rest"
            self.current_rep_angles = []
            self.current_rep_positions = []
        
        return self._get_status(angle)
    
    def _get_status(self, current_angle: Optional[float] = None) -> Dict:
        """Get current status dictionary"""
        return {
            'rep_count': self.rep_count,
            'state': self.current_state,
            'current_angle': current_angle,
            'frame_count': self.frame_count,
            'completed_reps': len(self.completed_reps)
        }
    
    def get_set_summary(self) -> SetMetrics:
        """Get comprehensive metrics for the completed set"""
        if not self.completed_reps:
            return SetMetrics(
                total_reps=0,
                valid_reps=0,
                invalid_reps=0,
                average_rom=0,
                average_form_score=0,
                average_tempo=0,
                consistency_score=0,
                total_duration=0,
                rep_details=[]
            )
        
        # Calculate aggregate metrics
        rom_values = [rep.range_of_motion for rep in self.completed_reps]
        form_scores = [rep.form_score for rep in self.completed_reps]
        durations = [rep.duration for rep in self.completed_reps]
        
        # Consistency score based on standard deviation of key metrics
        rom_std = np.std(rom_values)
        duration_std = np.std(durations)
        consistency = 100 - min(50, (rom_std + duration_std * 10))
        
        return SetMetrics(
            total_reps=self.rep_count,
            valid_reps=len(self.completed_reps),
            invalid_reps=self.rep_count - len(self.completed_reps),
            average_rom=np.mean(rom_values),
            average_form_score=np.mean(form_scores),
            average_tempo=np.mean(durations),
            consistency_score=max(0, consistency),
            total_duration=sum(durations),
            rep_details=self.completed_reps
        )
    
    def reset(self):
        """Reset counter for a new set"""
        self.rep_count = 0
        self.current_state = "rest"
        self.frame_count = 0
        self.angle_history.clear()
        self.current_rep_angles = []
        self.current_rep_positions = []
        self.completed_reps = []
        self.angle_buffer.clear()
        self.peak_angle = None
        self.bottom_angle = None
    
    def export_metrics(self, filepath: str):
        """Export detailed metrics to JSON"""
        set_summary = self.get_set_summary()
        
        export_data = {
            'exercise_type': self.exercise_type.value,
            'set_summary': {
                'total_reps': set_summary.total_reps,
                'valid_reps': set_summary.valid_reps,
                'invalid_reps': set_summary.invalid_reps,
                'average_rom': float(set_summary.average_rom),
                'average_form_score': float(set_summary.average_form_score),
                'average_tempo': float(set_summary.average_tempo),
                'consistency_score': float(set_summary.consistency_score),
                'total_duration': float(set_summary.total_duration)
            },
            'rep_details': [
                {
                    'rep_number': rep.rep_number,
                    'duration': rep.duration,
                    'range_of_motion': rep.range_of_motion,
                    'form_score': rep.form_score,
                    'tempo': rep.tempo,
                    'peak_angle': rep.peak_angle,
                    'bottom_angle': rep.bottom_angle,
                    'timestamp': rep.timestamp
                }
                for rep in set_summary.rep_details
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data


class MultiExerciseTracker:
    """
    Tracks multiple exercises in a workout session.
    Automatically switches between exercise types based on classification.
    """
    
    def __init__(self, fps: int = 30):
        self.fps = fps
        self.current_exercise: Optional[ExerciseType] = None
        self.current_counter: Optional[RepCounter] = None
        self.exercise_history: List[Tuple[ExerciseType, SetMetrics]] = []
        
    def update(self, landmarks: np.ndarray, exercise_type: ExerciseType, timestamp: float) -> Dict:
        """
        Update with new frame. Automatically handles exercise transitions.
        
        Args:
            landmarks: [33, 3] pose keypoints
            exercise_type: Detected exercise type
            timestamp: Current timestamp
            
        Returns:
            Status dictionary with rep counts and metrics
        """
        # Check if exercise changed
        if exercise_type != self.current_exercise:
            # Save previous exercise data if exists
            if self.current_counter is not None:
                set_metrics = self.current_counter.get_set_summary()
                self.exercise_history.append((self.current_exercise, set_metrics))
            
            # Start new exercise
            self.current_exercise = exercise_type
            self.current_counter = RepCounter(exercise_type, self.fps)
        
        # Update current exercise
        status = self.current_counter.update(landmarks, timestamp)
        status['exercise_type'] = exercise_type.value
        
        return status
    
    def get_workout_summary(self) -> Dict:
        """Get summary of entire workout session"""
        summary = {
            'total_exercises': len(self.exercise_history),
            'exercises': []
        }
        
        for exercise_type, set_metrics in self.exercise_history:
            summary['exercises'].append({
                'exercise': exercise_type.value,
                'total_reps': set_metrics.total_reps,
                'average_form_score': set_metrics.average_form_score,
                'total_duration': set_metrics.total_duration
            })
        
        return summary


if __name__ == "__main__":
    # Example usage
    print("=== Rep Counter Example ===\n")
    
    # Simulate squat exercise
    counter = RepCounter(ExerciseType.SQUAT, fps=30)
    
    # Simulate pose data for one rep (simplified)
    # In reality, this would come from MediaPipe
    
    print(f"Exercise: {counter.exercise_type.value}")
    print(f"Thresholds: {counter.thresholds}")
    print("\nSimulating rep counting...")
    print("Status: Ready to track reps")
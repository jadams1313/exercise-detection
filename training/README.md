# Dynamic Exercise Detection with Temporal GNN+LSTM

A real-time exercise detection system that uses Graph Neural Networks and LSTMs to recognize exercises from video with temporal awareness.

## 🎯 Overview

This project implements a two-stage deep learning architecture for exercise recognition:
- **Stage 1 (Spatial):** Graph Neural Networks extract pose features from each frame
- **Stage 2 (Temporal):** LSTM learns motion patterns across time

Unlike traditional approaches that classify entire videos, this model uses **sliding windows** for real-time detection capabilities.

## 🏗️ Architecture

```
Video Frames (30 frames)
    ↓
MediaPipe Pose (33 keypoints + orientation)
    ↓
Stage 1: Spatial GNN → Extract pose embeddings (64-dim per frame)
    ↓
Stage 2: Temporal LSTM → Learn motion patterns
    ↓
Classification → Exercise prediction
```

## 📊 Features

- **Window-based processing**: 30-frame sliding windows (~1 second at 30fps)
- **Real-time capable**: No need to upload full videos
- **Temporal awareness**: Understands motion patterns, not just static poses
- **Graph-based representation**: Naturally encodes skeletal structure
- **5 exercise types**: Barbell biceps curl, hammer curl, push-up, shoulder press, squat

## 🔬 Technical Deep Dive

### Part 1: Feature Extraction (MediaPipe)

#### Input
- Video frame: `H × W × 3` RGB image

#### Process
MediaPipe's pose detection CNN outputs:
- **33 keypoints** (nose, shoulders, elbows, wrists, hips, knees, ankles, etc.)
- Each keypoint: `(x, y, z)` coordinates

**Mathematical representation:**
```
Frame → CNN → P = {p₁, p₂, ..., p₃₃}
where pᵢ = (xᵢ, yᵢ, zᵢ) ∈ ℝ³
```

#### Orientation Calculation
Body orientation (roll, pitch, yaw) computed from keypoints:

```python
# Get shoulder and hip centers
shoulder_center = (left_shoulder + right_shoulder) / 2
hip_center = (left_hip + right_hip) / 2

# Create coordinate system
torso_vector = shoulder_center - hip_center      # Vertical axis
shoulder_vector = right_shoulder - left_shoulder  # Horizontal axis
forward_vector = torso_vector × shoulder_vector   # Cross product
```

These vectors form a rotation matrix:
```
R = [shoulder_vector | forward_vector | torso_vector]
```

Convert to Euler angles:
```
roll = atan2(R₃₂, R₃₃)
pitch = atan2(-R₃₁, sqrt(R₃₂² + R₃₃²))
yaw = atan2(R₂₁, R₁₁)
```

**Output per frame:**
```
[x₁, y₁, z₁, ..., x₃₃, y₃₃, z₃₃, roll, pitch, yaw, confidence]
└───────── 99 values ─────────┘  └──────── 4 values ────────┘
                        = 103 features total
```

### Part 2: Graph Construction

#### Why Graphs?
Body pose has inherent spatial structure - joints connect in specific ways (skeleton). Graphs naturally represent this topology.

#### Graph Definition
For each frame, we construct:

**Nodes:** 33 keypoints  
**Node features:** 7 dimensions per node
```
Node i: [xᵢ, yᵢ, zᵢ, roll, pitch, yaw, confidence]
        └─ position ─┘ └──── orientation ────┘
```

**Edges:** Connect keypoints following human skeleton
```
Examples:
- Shoulder (11) ↔ Elbow (13)
- Elbow (13) ↔ Wrist (15)
- Hip (23) ↔ Knee (25)
```

**Mathematical representation:**
```
Graph G = (V, E, X)
V = {v₁, ..., v₃₃}              # 33 nodes
E = {(vᵢ, vⱼ) | i,j connected}  # Edges (skeleton connections)
X ∈ ℝ³³ˣ⁷                       # Node feature matrix
```

### Part 3: Stage 1 - Spatial GNN

#### Goal
Extract spatial features from each pose graph independently.

#### Graph Convolutional Network (GCN)

**Core Operation:**
```
H⁽ˡ⁺¹⁾ = σ(D̃⁻½ Ã D̃⁻½ H⁽ˡ⁾ W⁽ˡ⁾)
```

Where:
- `H⁽ˡ⁾`: Node features at layer l
- `Ã = A + I`: Adjacency matrix with self-loops
- `D̃`: Degree matrix (diagonal)
- `W⁽ˡ⁾`: Learnable weight matrix
- `σ`: Activation function (ReLU)

**What this does:**  
Each node aggregates information from its neighbors. For example, the elbow node combines features from its own position plus shoulder and wrist features.

#### Our 3-Layer Architecture

**Layer 1:**
```
Input:  X ∈ ℝ³³ˣ⁷    (33 nodes, 7 features each)
Output: H⁽¹⁾ ∈ ℝ³³ˣ¹²⁸  (128-dim features per node)
```

**Layer 2:**
```
Input:  H⁽¹⁾ ∈ ℝ³³ˣ¹²⁸
Output: H⁽²⁾ ∈ ℝ³³ˣ¹²⁸
```

**Layer 3:**
```
Input:  H⁽²⁾ ∈ ℝ³³ˣ¹²⁸
Output: H⁽³⁾ ∈ ℝ³³ˣ⁶⁴   (64-dim features per node)
```

#### Global Pooling

After 3 GCN layers, we have 33 nodes with 64 features each. We need a single vector representing the entire pose.

**Mean pooling:**
```
h = (1/33) Σᵢ₌₁³³ hᵢ⁽³⁾
```

**Result:** `h ∈ ℝ⁶⁴` - single 64-dimensional pose embedding

**Why mean pooling?** It's translation-invariant and captures average body configuration.

### Part 4: Stage 2 - Temporal LSTM

#### Goal
Learn motion patterns across the sequence of poses.

#### Input Sequence
After processing 30 frames through the GNN:
```
Window = [h₁, h₂, ..., h₃₀]  where hₜ ∈ ℝ⁶⁴
```

#### LSTM Cell Mathematics

The LSTM uses 4 gates to control information flow at each timestep:

**1. Forget gate** (what to forget from memory):
```
fₜ = σ(Wf · [hₜ₋₁, xₜ] + bf)
```

**2. Input gate** (what new information to add):
```
iₜ = σ(Wi · [hₜ₋₁, xₜ] + bi)
C̃ₜ = tanh(Wc · [hₜ₋₁, xₜ] + bc)
```

**3. Cell state update** (update memory):
```
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ
```
- `⊙`: Element-wise multiplication
- Forget old: `fₜ ⊙ Cₜ₋₁`
- Add new: `iₜ ⊙ C̃ₜ`

**4. Output gate** (what to output):
```
oₜ = σ(Wo · [hₜ₋₁, xₜ] + bo)
hₜ = oₜ ⊙ tanh(Cₜ)
```

**Intuition:**  
The LSTM maintains a "memory" of motion over time. At each frame, it decides what's important to remember (e.g., "arm is moving upward") and what to forget (e.g., "previous stance irrelevant").

#### Our 2-Layer LSTM

**Layer 1:**
```
For t = 1 to 30:
    h₁,ₜ = LSTM₁(pose_embedding_t, h₁,ₜ₋₁)
```

**Layer 2:**
```
For t = 1 to 30:
    h₂,ₜ = LSTM₂(h₁,ₜ, h₂,ₜ₋₁)
```

**Output:**
```
h₂,₃₀ ∈ ℝ¹²⁸  (final hidden state)
```

### Part 5: Classification

#### Final MLP (Multi-Layer Perceptron)

**Layer 1:**
```
z = ReLU(W₁ · h₂,₃₀ + b₁)
z ∈ ℝ⁶⁴
```

**Dropout:**
```
z = Dropout(z, p=0.5)  # Randomly zero 50% during training
```
*Purpose: Prevents overfitting*

**Layer 2:**
```
logits = W₂ · z + b₂
logits ∈ ℝ⁵  (5 exercise classes)
```

**Softmax:**
```
P(class = i) = exp(logitsᵢ) / Σⱼ exp(logitsⱼ)
```
*Converts logits to probabilities that sum to 1*

**Prediction:**
```
predicted_class = argmax(P)
```

### Part 6: Training

#### Loss Function: Cross-Entropy

```
L = -Σᵢ yᵢ log(ŷᵢ)
```

Where:
- `yᵢ`: True label (one-hot encoded)
- `ŷᵢ`: Predicted probability

**Example:**
```
True label: "bicep curl" → [0, 1, 0, 0, 0]
Predicted:                  [0.05, 0.80, 0.10, 0.03, 0.02]

Loss = -log(0.80) = 0.223
```

#### Optimization: Adam

**Gradient descent update:**
```
W ← W - η · ∂L/∂W
```

Where:
- `η`: Learning rate (0.001)
- `∂L/∂W`: Gradient computed via backpropagation

**Adam optimizer** enhances this with:
1. Momentum (average of past gradients)
2. Adaptive learning rates per parameter
3. Faster convergence

### Part 7: Complete Forward Pass

Let's trace one 30-frame window through the entire model:

#### Frame 1
```
1. MediaPipe: [720×1280 image] → [103 features]
2. Graph: Create 33-node graph with 7 features/node
3. GCN: 
   - Layer 1: ℝ³³ˣ⁷ → ℝ³³ˣ¹²⁸
   - Layer 2: ℝ³³ˣ¹²⁸ → ℝ³³ˣ¹²⁸
   - Layer 3: ℝ³³ˣ¹²⁸ → ℝ³³ˣ⁶⁴
4. Pool: ℝ³³ˣ⁶⁴ → ℝ⁶⁴
Result: h₁ ∈ ℝ⁶⁴
```

#### Frames 2-30
Same process → h₂, ..., h₃₀

#### Sequence Processing
```
Stack: [h₁, h₂, ..., h₃₀] → ℝ³⁰ˣ⁶⁴
LSTM: ℝ³⁰ˣ⁶⁴ → h_final ∈ ℝ¹²⁸
```

#### Classification
```
MLP: h_final → ℝ¹²⁸ → ℝ⁶⁴ → ℝ⁵
Softmax: [0.05, 0.85, 0.05, 0.03, 0.02]
Prediction: Class 1 (bicep curl)
```

## 🚀 Getting Started

### Installation

```bash
pip install torch torchvision
pip install torch-geometric
pip install mediapipe
pip install opencv-python
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Quick Start

```python
from training.train_temporal import TemporalExerciseTrainer

# Initialize trainer
trainer = TemporalExerciseTrainer("path/to/dataset")

# Create windowed dataset
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
```

### Dataset Structure

```
dataset/
├── exercise_1/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
├── exercise_2/
│   └── ...
└── processed/
    └── pose_features.csv  (generated automatically)
```

## 📈 Performance

- **Accuracy**: Achieves competitive accuracy on 5 exercise types
- **Speed**: Real-time capable with 30-frame windows
- **Data efficiency**: Sliding windows generate multiple training samples per video

## 🔑 Key ML Principles

1. **Inductive Bias**: Graph structure encodes skeletal relationships
2. **Hierarchical Learning**: GCN learns spatial patterns, LSTM learns temporal patterns
3. **Representation Learning**: Raw pixels → keypoints → graphs → embeddings → predictions
4. **End-to-End Training**: All components trained together, gradients flow through entire pipeline
5. **Regularization**: Dropout prevents overfitting

## 📁 Project Structure

```
exercise-detection/
├── training/
│   ├── load_data.py          # Dataset loading utilities
│   ├── train.py              # Original static model
│   └── train_temporal.py     # Temporal GNN+LSTM model
├── temporal_exercise_demo.ipynb  # Demo notebook
└── README.md
```

## 🎯 Next Steps

- [ ] Real-time webcam inference
- [ ] Rep counting logic
- [ ] Form quality assessment
- [ ] Performance analytics dashboard
- [ ] Mobile deployment

## 📚 References

- MediaPipe Pose: [Google MediaPipe](https://google.github.io/mediapipe/)
- Graph Convolutional Networks: [Kipf & Welling (2017)](https://arxiv.org/abs/1609.02907)
- LSTM Networks: [Hochreiter & Schmidhuber (1997)](https://www.bioinf.jku.at/publications/older/2604.pdf)

## 📝 License

MIT License

## 🤝 Contributing

Contributions welcome! Please feel free to submit a Pull Request.

---

**Built with PyTorch, PyTorch Geometric, and MediaPipe**

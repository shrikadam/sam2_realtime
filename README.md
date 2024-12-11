# SAM2 Real-Time Detection and Tracking

This repository adapts **SAM2** to include real-time multi-object tracking. It allows users to specify and track a 
fixed number of objects in real time, integrating motion-aware memory selection from **SAMURAI** for improved tracking 
in complex scenarios.  

### About SAM2
**SAM2** (Segment Anything Model 2) is designed for object segmentation and tracking but lacks built-in capabilities 
for performing this in real time.

### About SAMURAI
**SAMURAI** enhances SAM2 by introducing motion-aware memory selection, leveraging temporal motion cues for better 
tracking accuracy without retraining or fine-tuning.  

## Key Features

- **Real-Time Tracking**: Modified SAM2 to track a fixed number of objects in real time.
- **Motion-Aware Memory**: SAMURAI leverages temporal motion cues for robust object tracking without retraining or fine-tuning.
- **YOLO Integration**: Utilizes YOLO for object detection and mask generation as input to SAM2.

The core implementation resides in `sam2_object_tracker.py`, where the number of objects to track must be specified during instantiation.

---

## Setup Instructions

### 1. Create Conda Environment
```bash
conda env create -f environment.yml
```

### 2. Install SAM2
```bash
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```


### 4. Download SAM2 Checkpoints
```bash
cd checkpoints
./download_ckpts.sh
cd ..
```

---

## Usage

### Demo
Run the demo notebook to visualize YOLO object detection and SAM2 object tracking in action:  
`scripts/demo.ipynb`

### Inference
To perform detection and tracking on a video source, use the following script:  
```bash
cd scripts/
python detect_and_track.py \
  --source "/data/datasets/SAM2/sav_train/sav_021/sav_021835.mp4" \
  --num_objects 1 \
  --visualize \
  --device "cuda:0"
```

---

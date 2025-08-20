# Precision Farming Robot - Weed Detection with YOLO

## Project Overview
This project implements weed detection using YOLO (You Only Look Once) for a precision farming robot. The system can identify and locate weeds in agricultural images, enabling targeted weed elimination.

## Project Structure
```
├── data/                   # Dataset management
│   ├── images/            # Training and validation images
│   ├── labels/            # YOLO format annotations
│   └── dataset.yaml       # Dataset configuration
├── models/                # Trained models and checkpoints
├── training/              # Training scripts and configurations
├── inference/             # Inference and prediction scripts
├── utils/                 # Utility functions
├── web_app/               # Web application (future mobile conversion)
└── requirements.txt       # Python dependencies
```

## Features
- **Real-time Weed Detection**: Detect weeds in images using YOLO
- **Bounding Box Visualization**: Draw boxes around detected weeds
- **Confidence Scoring**: Provide confidence levels for detections
- **Web Interface**: User-friendly interface for image upload and detection
- **Mobile-Ready**: Structured for easy conversion to mobile app

## Getting Started

### Prerequisites
- Python 3.8+
- Virtual environment (already set up)
- GPU recommended for training (optional)

### Installation
1. Activate the virtual environment:
   ```bash
   # Windows
   yolo-weed-env\Scripts\activate
   
   # Linux/Mac
   source yolo-weed-env/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. **Data Preparation**: Place weed images in `data/images/` and annotations in `data/labels/`
2. **Training**: Run training script to train the YOLO model
3. **Inference**: Use trained model to detect weeds in new images
4. **Web App**: Launch web interface for interactive weed detection

## Next Steps
1. Collect weed dataset
2. Annotate images with weed locations
3. Train YOLO model
4. Test detection accuracy
5. Integrate with robot hardware

## Technology Stack
- **YOLO**: Object detection model
- **PyTorch**: Deep learning framework
- **OpenCV**: Image processing
- **Flask**: Web application framework
- **PostgreSQL**: Database (Neon cloud)

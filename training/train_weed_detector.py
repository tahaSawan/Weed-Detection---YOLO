"""
YOLO Weed Detection Training Script
This script trains a YOLO model to detect weeds in agricultural images.
"""

import os
import yaml
from ultralytics import YOLO
import torch

def create_dataset_config():
    """
    Create the dataset configuration file for YOLO training.
    This tells YOLO where to find your images and labels.
    """
    dataset_config = {
        'path': '../data',  # Path to your dataset
        'train': 'images/train',  # Training images
        'val': 'images/val',      # Validation images
        'test': 'images/test',    # Test images (optional)
        
        # Classes (weed types you want to detect)
        'names': {
            0: 'weed',           # General weed class
            # Add more classes as needed:
            # 1: 'dandelion',
            # 2: 'thistle',
            # 3: 'crabgrass'
        },
        
        # Number of classes
        'nc': 1
    }
    
    # Save the configuration
    with open('../data/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("✅ Dataset configuration created at data/dataset.yaml")
    return dataset_config

def setup_training_environment():
    """
    Check if your system is ready for training.
    """
    print("🔍 Checking training environment...")
    
    # Check if CUDA (GPU) is available
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        device = 'cuda'
    else:
        print("⚠️  GPU not available, using CPU (training will be slower)")
        device = 'cpu'
    
    # Check if dataset directory exists
    if not os.path.exists('../data/images'):
        print("❌ Dataset directory not found. Please create data/images/ directory")
        return False
    
    print("✅ Training environment ready!")
    return True

def train_model(epochs=100, batch_size=16, img_size=640):
    """
    Train the YOLO model for weed detection.
    
    Args:
        epochs: Number of training epochs
        batch_size: Number of images per batch
        img_size: Input image size
    """
    print("🚀 Starting YOLO training...")
    
    # Load a pre-trained YOLO model
    # YOLOv8n is the smallest and fastest model
    model = YOLO('yolov8n.pt')  # Load pre-trained model
    
    # Start training
    results = model.train(
        data='../data/dataset.yaml',  # Dataset configuration
        epochs=epochs,                # Number of epochs
        batch=batch_size,             # Batch size
        imgsz=img_size,               # Image size
        device='auto',                # Use GPU if available
        project='../models',          # Save results to models/
        name='weed_detector',         # Experiment name
        patience=20,                  # Early stopping patience
        save=True,                    # Save best model
        verbose=True                  # Show training progress
    )
    
    print("✅ Training completed!")
    print(f"📁 Model saved in: models/weed_detector/")
    
    return results

def validate_model():
    """
    Validate the trained model on test data.
    """
    print("🔍 Validating model...")
    
    # Load the best trained model
    model = YOLO('../models/weed_detector/weights/best.pt')
    
    # Run validation
    results = model.val()
    
    print("✅ Validation completed!")
    print(f"📊 mAP50: {results.box.map50:.3f}")
    print(f"📊 mAP50-95: {results.box.map:.3f}")
    
    return results

def main():
    """
    Main function to run the complete training pipeline.
    """
    print("🌱 YOLO Weed Detection Training Pipeline")
    print("=" * 50)
    
    # Step 1: Create dataset configuration
    print("\n📋 Step 1: Creating dataset configuration...")
    create_dataset_config()
    
    # Step 2: Check environment
    print("\n🔍 Step 2: Checking training environment...")
    if not setup_training_environment():
        print("❌ Environment setup failed. Please check the errors above.")
        return
    
    # Step 3: Train model
    print("\n🚀 Step 3: Training YOLO model...")
    print("⚠️  Note: This will take some time depending on your dataset size and hardware.")
    
    # You can adjust these parameters based on your needs
    train_model(
        epochs=50,      # Start with fewer epochs for testing
        batch_size=8,   # Smaller batch size if you have limited memory
        img_size=640    # Standard YOLO image size
    )
    
    # Step 4: Validate model
    print("\n🔍 Step 4: Validating trained model...")
    validate_model()
    
    print("\n🎉 Training pipeline completed!")
    print("\n📁 Your trained model is saved in: models/weed_detector/weights/best.pt")
    print("🔧 Next steps:")
    print("   1. Test the model on new images")
    print("   2. Adjust training parameters if needed")
    print("   3. Collect more data if accuracy is low")
    print("   4. Integrate with your robot system")

if __name__ == "__main__":
    main()

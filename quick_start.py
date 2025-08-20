#!/usr/bin/env python3
"""
Quick Start Script for YOLO Weed Detection Project
This script helps you get started with the weed detection project.
"""

import os
import sys
import subprocess
import webbrowser
from pathlib import Path

def print_header():
    """Print project header."""
    print("🌱" + "="*60 + "🌱")
    print("    YOLO Weed Detection - Precision Farming Robot")
    print("🌱" + "="*60 + "🌱")
    print()

def check_environment():
    """Check if the environment is properly set up."""
    print("🔍 Checking environment...")
    
    # Check if virtual environment is activated
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("⚠️  Virtual environment not detected.")
        print("   Please activate your virtual environment first:")
        print("   Windows: yolo-weed-env\\Scripts\\activate")
        print("   Linux/Mac: source yolo-weed-env/bin/activate")
        print()
        return False
    
    print("✅ Virtual environment is active")
    
    # Check if required packages are installed
    try:
        import torch
        import ultralytics
        import cv2
        import flask
        print("✅ Required packages are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("   Please install requirements: pip install -r requirements.txt")
        return False

def create_sample_dataset():
    """Create sample dataset structure."""
    print("\n📁 Creating sample dataset structure...")
    
    # Create directories
    directories = [
        'data/images/train',
        'data/images/val',
        'data/images/test',
        'data/labels/train',
        'data/labels/val',
        'data/labels/test',
        'inference/results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"   ✅ Created: {directory}")
    
    # Create sample dataset.yaml
    dataset_config = {
        'path': './data',
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {
            0: 'weed'
        },
        'nc': 1
    }
    
    import yaml
    with open('data/dataset.yaml', 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print("   ✅ Created: data/dataset.yaml")
    print("\n📋 Next steps:")
    print("   1. Add weed images to data/images/train/ and data/images/val/")
    print("   2. Annotate images using LabelImg or Roboflow")
    print("   3. Place annotation files in data/labels/train/ and data/labels/val/")

def show_menu():
    """Show the main menu."""
    print("\n🚀 What would you like to do?")
    print("1. 📊 Train YOLO model")
    print("2. 🔍 Test weed detection on image")
    print("3. 🎥 Test weed detection on video")
    print("4. 📹 Real-time detection with camera")
    print("5. 🌐 Launch web application")
    print("6. 📁 Setup dataset structure")
    print("7. 📖 View project documentation")
    print("8. ❌ Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            if choice in ['1', '2', '3', '4', '5', '6', '7', '8']:
                return choice
            else:
                print("❌ Please enter a number between 1 and 8")
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)

def train_model():
    """Start model training."""
    print("\n🚀 Starting YOLO training...")
    print("⚠️  Make sure you have:")
    print("   - Images in data/images/train/ and data/images/val/")
    print("   - Annotations in data/labels/train/ and data/labels/val/")
    
    confirm = input("\nContinue with training? (y/n): ").lower().strip()
    if confirm == 'y':
        try:
            subprocess.run([sys.executable, 'training/train_weed_detector.py'], check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Training failed: {e}")
        except FileNotFoundError:
            print("❌ Training script not found. Please check the file path.")

def test_image_detection():
    """Test weed detection on an image."""
    print("\n🔍 Testing weed detection on image...")
    
    image_path = input("Enter path to image file: ").strip()
    if not os.path.exists(image_path):
        print("❌ Image file not found")
        return
    
    try:
        subprocess.run([
            sys.executable, 'inference/detect_weeds.py',
            '--image', image_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection failed: {e}")
    except FileNotFoundError:
        print("❌ Detection script not found. Please check the file path.")

def test_video_detection():
    """Test weed detection on a video."""
    print("\n🎥 Testing weed detection on video...")
    
    video_path = input("Enter path to video file: ").strip()
    if not os.path.exists(video_path):
        print("❌ Video file not found")
        return
    
    try:
        subprocess.run([
            sys.executable, 'inference/detect_weeds.py',
            '--video', video_path
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Detection failed: {e}")
    except FileNotFoundError:
        print("❌ Detection script not found. Please check the file path.")

def real_time_detection():
    """Start real-time detection with camera."""
    print("\n📹 Starting real-time weed detection...")
    print("Press 'q' to quit the detection window")
    
    try:
        subprocess.run([
            sys.executable, 'inference/detect_weeds.py',
            '--camera'
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Real-time detection failed: {e}")
    except FileNotFoundError:
        print("❌ Detection script not found. Please check the file path.")

def launch_web_app():
    """Launch the web application."""
    print("\n🌐 Launching web application...")
    print("The web app will open in your browser at: http://localhost:5000")
    
    try:
        # Start the web app in a separate process
        process = subprocess.Popen([
            sys.executable, 'web_app/app.py'
        ])
        
        # Wait a moment for the server to start
        import time
        time.sleep(3)
        
        # Open browser
        webbrowser.open('http://localhost:5000')
        
        print("✅ Web application started!")
        print("Press Ctrl+C to stop the server")
        
        try:
            process.wait()
        except KeyboardInterrupt:
            process.terminate()
            print("\n🛑 Web application stopped")
            
    except FileNotFoundError:
        print("❌ Web app script not found. Please check the file path.")
    except Exception as e:
        print(f"❌ Failed to start web app: {e}")

def show_documentation():
    """Show project documentation."""
    print("\n📖 Project Documentation:")
    print("="*50)
    
    docs = [
        ("README.md", "Project overview and setup instructions"),
        ("YOLO_WEED_DETECTION_GUIDE.md", "Complete YOLO guide and workflow"),
    ]
    
    for doc, description in docs:
        if os.path.exists(doc):
            print(f"📄 {doc}: {description}")
        else:
            print(f"❌ {doc}: Not found")
    
    print("\n🔗 Useful Resources:")
    print("   - YOLO Documentation: https://docs.ultralytics.com/")
    print("   - LabelImg (Annotation Tool): https://github.com/tzutalin/labelImg")
    print("   - Roboflow (Online Annotation): https://roboflow.com/")
    
    # Try to open README in default text editor
    if os.path.exists("README.md"):
        try:
            if sys.platform == "win32":
                os.startfile("README.md")
            elif sys.platform == "darwin":
                subprocess.run(["open", "README.md"])
            else:
                subprocess.run(["xdg-open", "README.md"])
        except:
            pass

def main():
    """Main function."""
    print_header()
    
    # Check environment
    if not check_environment():
        print("\n❌ Environment setup incomplete. Please fix the issues above.")
        return
    
    # Show menu and handle choices
    while True:
        choice = show_menu()
        
        if choice == '1':
            train_model()
        elif choice == '2':
            test_image_detection()
        elif choice == '3':
            test_video_detection()
        elif choice == '4':
            real_time_detection()
        elif choice == '5':
            launch_web_app()
        elif choice == '6':
            create_sample_dataset()
        elif choice == '7':
            show_documentation()
        elif choice == '8':
            print("\n👋 Goodbye! Happy weed detecting!")
            break
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()

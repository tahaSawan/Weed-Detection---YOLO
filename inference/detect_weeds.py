"""
Weed Detection Inference Script
This script uses a trained YOLO model to detect weeds in new images.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import os
import argparse
from pathlib import Path

class WeedDetector:
    def __init__(self, model_path='../models/weed_detector/weights/best.pt'):
        """
        Initialize the weed detector with a trained YOLO model.
        
        Args:
            model_path: Path to the trained YOLO model
        """
        self.model = YOLO(model_path)
        print(f"✅ Loaded model from: {model_path}")
    
    def detect_weeds(self, image_path, confidence=0.5, save_result=True):
        """
        Detect weeds in an image.
        
        Args:
            image_path: Path to the image file
            confidence: Minimum confidence threshold (0-1)
            save_result: Whether to save the result image
        
        Returns:
            List of detections with bounding boxes and confidence scores
        """
        print(f"🔍 Detecting weeds in: {image_path}")
        
        # Run inference
        results = self.model(image_path, conf=confidence)
        
        # Get the first result (assuming single image)
        result = results[0]
        
        # Extract detections
        detections = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes
            confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
            class_ids = result.boxes.cls.cpu().numpy()  # Class IDs
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                detection = {
                    'bbox': box.tolist(),  # [x1, y1, x2, y2]
                    'confidence': float(conf),
                    'class_id': int(class_id),
                    'class_name': 'weed'  # You can map this to actual class names
                }
                detections.append(detection)
        
        print(f"✅ Found {len(detections)} weed(s)")
        
        # Save result image if requested
        if save_result:
            self.save_result_image(image_path, result)
        
        return detections
    
    def save_result_image(self, image_path, result):
        """
        Save the detection result as an image with bounding boxes.
        """
        # Create output directory
        output_dir = Path('../inference/results')
        output_dir.mkdir(exist_ok=True)
        
        # Generate output filename
        input_name = Path(image_path).stem
        output_path = output_dir / f"{input_name}_detected.jpg"
        
        # Save the result
        result.save(str(output_path))
        print(f"📁 Result saved to: {output_path}")
    
    def detect_video(self, video_path, confidence=0.5, output_path=None):
        """
        Detect weeds in a video file.
        
        Args:
            video_path: Path to the video file
            confidence: Minimum confidence threshold
            output_path: Path to save the output video
        """
        print(f"🎥 Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Setup video writer
        if output_path is None:
            output_path = f"../inference/results/{Path(video_path).stem}_detected.mp4"
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection on frame
            results = self.model(frame, conf=confidence)
            result = results[0]
            
            # Draw results on frame
            annotated_frame = result.plot()
            
            # Write frame to output video
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Print progress every 30 frames
                print(f"📹 Processed {frame_count} frames...")
        
        # Clean up
        cap.release()
        out.release()
        print(f"✅ Video processing completed. Output saved to: {output_path}")
    
    def real_time_detection(self, camera_id=0, confidence=0.5):
        """
        Real-time weed detection using webcam.
        
        Args:
            camera_id: Camera device ID (usually 0 for default camera)
            confidence: Minimum confidence threshold
        """
        print("📹 Starting real-time weed detection...")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Error: Could not open camera {camera_id}")
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Run detection
            results = self.model(frame, conf=confidence)
            result = results[0]
            
            # Draw results
            annotated_frame = result.plot()
            
            # Display frame
            cv2.imshow('Weed Detection - Real Time', annotated_frame)
            
            # Check for quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Real-time detection stopped")

def main():
    """
    Main function to run weed detection.
    """
    parser = argparse.ArgumentParser(description='Weed Detection with YOLO')
    parser.add_argument('--model', type=str, 
                       default='../models/weed_detector/weights/best.pt',
                       help='Path to trained YOLO model')
    parser.add_argument('--image', type=str, help='Path to image file')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--camera', action='store_true', help='Use webcam for real-time detection')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold (0-1)')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        print("Please train a model first using the training script.")
        return
    
    # Initialize detector
    detector = WeedDetector(args.model)
    
    # Run detection based on input type
    if args.image:
        detections = detector.detect_weeds(args.image, args.confidence)
        print(f"\n📊 Detection Results:")
        for i, detection in enumerate(detections):
            print(f"  Weed {i+1}: Confidence {detection['confidence']:.2f}")
    
    elif args.video:
        detector.detect_video(args.video, args.confidence)
    
    elif args.camera:
        detector.real_time_detection(confidence=args.confidence)
    
    else:
        print("❌ Please specify --image, --video, or --camera")
        print("\nExample usage:")
        print("  python detect_weeds.py --image path/to/image.jpg")
        print("  python detect_weeds.py --video path/to/video.mp4")
        print("  python detect_weeds.py --camera")

if __name__ == "__main__":
    main()

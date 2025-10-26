"""
DAY 1: Basic Object Detection System
File: basic_detector.py

This is your first working object detection code!
"""

from ultralytics import YOLO
import cv2
import os

class ObjectDetector:
    def __init__(self):
        """Initialize the detector with YOLOv8 model"""
        print("🔄 Loading YOLOv8 model...")
        print("⏳ First time? Model will download (~6MB)...")
        
        # Load pre-trained YOLOv8 nano model (fastest)
        self.model = YOLO('yolov8n.pt')
        
        print("✅ Model loaded successfully!")
        print(f"📊 Can detect {len(self.model.names)} object classes")
        
    def detect_image(self, image_path, save_path='output_detected.jpg'):
        """
        Detect objects in an image
        
        Args:
            image_path: Path to input image
            save_path: Path to save output image
        """
        print(f"\n🔍 Analyzing image: {image_path}")
        
        # Run detection
        results = self.model(image_path)
        result = results[0]
        
        # Get detection info
        num_objects = len(result.boxes)
        print(f"\n✅ Found {num_objects} objects!")
        
        # Print each detected object
        detected_objects = {}
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = self.model.names[class_id]
            
            # Count objects
            detected_objects[class_name] = detected_objects.get(class_name, 0) + 1
            
            print(f"   📌 {class_name}: {confidence*100:.1f}% confidence")
        
        # Summary
        print(f"\n📊 Summary:")
        for obj, count in detected_objects.items():
            print(f"   • {obj}: {count}")
        
        # Save result with bounding boxes
        result.save(save_path)
        print(f"\n💾 Saved result to: {save_path}")
        
        return result, detected_objects
    
    def detect_video(self, video_path, output_path='output_video.mp4'):
        """
        Detect objects in a video file
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video
        """
        print(f"\n🎥 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"📊 Video Info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Write frame
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Progress every 30 frames
                progress = (frame_count / total_frames) * 100
                print(f"⏳ Progress: {progress:.1f}%")
        
        cap.release()
        out.release()
        
        print(f"\n✅ Video processing complete!")
        print(f"💾 Saved to: {output_path}")
    
    def detect_webcam(self):
        """
        Real-time detection from webcam
        Press 'q' to quit
        """
        print("\n📹 Starting webcam...")
        print("⌨️  Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam!")
            return
        
        print("✅ Webcam started! Detection running...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error reading frame")
                break
            
            # Run detection
            results = self.model(frame, verbose=False)
            annotated_frame = results[0].plot()
            
            # Display
            cv2.imshow('Object Detection - Press Q to Quit', annotated_frame)
            
            # Quit on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Webcam stopped")


def main():
    """Main function - Menu driven program"""
    
    print("=" * 60)
    print("🎯 OBJECT DETECTION SYSTEM - DAY 1")
    print("=" * 60)
    
    # Initialize detector
    detector = ObjectDetector()
    
    while True:
        print("\n" + "=" * 60)
        print("📋 MENU:")
        print("=" * 60)
        print("1. Detect from Image (online sample)")
        print("2. Detect from Your Image")
        print("3. Detect from Video File")
        print("4. Real-time Webcam Detection")
        print("5. Exit")
        print("=" * 60)
        
        choice = input("\n👉 Enter your choice (1-5): ")
        
        if choice == '1':
            # Use online sample image
            print("\n📥 Using sample image from internet...")
            sample_url = 'https://ultralytics.com/images/bus.jpg'
            detector.detect_image(sample_url)
            print("\n✅ Check 'output_detected.jpg' in your project folder!")
            
        elif choice == '2':
            image_path = input("\n📁 Enter image path: ")
            if os.path.exists(image_path):
                detector.detect_image(image_path)
                print("\n✅ Check 'output_detected.jpg' in your project folder!")
            else:
                print("❌ Error: File not found!")
        
        elif choice == '3':
            video_path = input("\n📁 Enter video path: ")
            if os.path.exists(video_path):
                detector.detect_video(video_path)
                print("\n✅ Check 'output_video.mp4' in your project folder!")
            else:
                print("❌ Error: File not found!")
        
        elif choice == '4':
            detector.detect_webcam()
        
        elif choice == '5':
            print("\n👋 Goodbye! Happy Coding!")
            break
        
        else:
            print("❌ Invalid choice! Please enter 1-5")


if __name__ == "__main__":
    main()
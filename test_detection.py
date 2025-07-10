#!/usr/bin/env python3
"""
Test YOLO detection on a single frame to debug what's being detected
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_yolo_detection():
    """Test YOLO detection on a sample frame"""
    
    # Load model
    model = YOLO("models/yolo_model.pt")
    
    # Load a sample frame from the video
    cap = cv2.VideoCapture("input/15sec_input_720p.mp4")
    
    # Skip to middle of video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ret, frame = cap.read()
    
    if not ret:
        print("Could not read frame")
        return
    
    print(f"Frame shape: {frame.shape}")
    
    # Run detection with very low confidence to see everything
    results = model(frame, verbose=True, conf=0.1)
    
    print(f"Found {len(results)} result objects")
    
    for i, result in enumerate(results):
        print(f"Result {i}:")
        boxes = result.boxes
        if boxes is not None:
            print(f"  Found {len(boxes)} boxes")
            
            # Get all classes and their names
            classes = boxes.cls.cpu().numpy()
            confidences = boxes.conf.cpu().numpy()
            boxes_xyxy = boxes.xyxy.cpu().numpy()
            
            print(f"  Classes: {classes}")
            print(f"  Confidences: {confidences}")
            
            # Print class names
            for j, (cls, conf, box) in enumerate(zip(classes, confidences, boxes_xyxy)):
                class_name = model.names[int(cls)]
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                rel_width = width / frame.shape[1]
                rel_height = height / frame.shape[0]
                aspect_ratio = height / max(width, 1)
                
                print(f"    Detection {j}: {class_name} (conf={conf:.3f})")
                print(f"      Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
                print(f"      Size: {width:.0f}x{height:.0f} ({rel_width:.3f}x{rel_height:.3f})")
                print(f"      Aspect ratio: {aspect_ratio:.2f}")
                print()
    
    # Draw results
    annotated_frame = results[0].plot()
    
    # Save annotated frame
    cv2.imwrite("output/detection_test.jpg", annotated_frame)
    cv2.imshow("Annotated Frame", annotated_frame)
    print("Saved annotated frame to output/detection_test.jpg")
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cap.release()

if __name__ == "__main__":
    test_yolo_detection() 
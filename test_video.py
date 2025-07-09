#!/usr/bin/env python3
"""
Video Testing and Conversion Script
This script helps test video playback and converts to multiple formats
"""

import cv2
import os
import subprocess
import sys

def test_video_playback(video_path):
    """Test if video can be opened and played with OpenCV"""
    print(f"Testing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return False
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return False
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"✅ Video properties:")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps}")
    print(f"   Frames: {frame_count}")
    print(f"   Duration: {frame_count/fps:.2f} seconds")
    
    # Test reading a few frames
    frames_read = 0
    for i in range(min(10, frame_count)):
        ret, frame = cap.read()
        if ret:
            frames_read += 1
        else:
            break
    
    cap.release()
    
    if frames_read > 0:
        print(f"✅ Successfully read {frames_read} frames")
        return True
    else:
        print(f"❌ Could not read any frames")
        return False

def convert_to_formats(input_video, output_dir="output"):
    """Convert video to multiple formats for compatibility"""
    if not os.path.exists(input_video):
        print(f"❌ Input video not found: {input_video}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(input_video))[0]
    
    formats = [
        {
            'name': 'H.264 MP4 (Most Compatible)',
            'output': f"{output_dir}/{base_name}_h264.mp4",
            'cmd': ['ffmpeg', '-i', input_video, '-c:v', 'libx264', '-crf', '23', '-preset', 'medium', '-y']
        },
        {
            'name': 'H.265 MP4 (Smaller Size)',
            'output': f"{output_dir}/{base_name}_h265.mp4",
            'cmd': ['ffmpeg', '-i', input_video, '-c:v', 'libx265', '-crf', '28', '-preset', 'medium', '-y']
        },
        {
            'name': 'WebM (Web Compatible)',
            'output': f"{output_dir}/{base_name}.webm",
            'cmd': ['ffmpeg', '-i', input_video, '-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0', '-y']
        },
        {
            'name': 'AVI (Legacy Compatible)',
            'output': f"{output_dir}/{base_name}.avi",
            'cmd': ['ffmpeg', '-i', input_video, '-c:v', 'libx264', '-crf', '23', '-y']
        }
    ]
    
    print(f"\n🔄 Converting {input_video} to multiple formats...")
    
    for fmt in formats:
        print(f"\n📹 Creating {fmt['name']}...")
        cmd = fmt['cmd'] + [fmt['output']]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"✅ Successfully created: {fmt['output']}")
                # Test the converted file
                test_video_playback(fmt['output'])
            else:
                print(f"❌ Failed to create {fmt['name']}")
                print(f"Error: {result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"❌ Timeout while creating {fmt['name']}")
        except Exception as e:
            print(f"❌ Error creating {fmt['name']}: {e}")

def main():
    """Main function"""
    print("🎬 Video Testing and Conversion Tool")
    print("=" * 50)
    
    # Test original video
    original_video = "output/tracked_output.mp4"
    h264_video = "output/tracked_output_h264.mp4"
    
    print("\n1. Testing original video:")
    test_video_playback(original_video)
    
    print("\n2. Testing H.264 converted video:")
    test_video_playback(h264_video)
    
    # Convert to multiple formats
    print("\n3. Converting to multiple formats for maximum compatibility:")
    convert_to_formats(original_video)
    
    print("\n" + "=" * 50)
    print("🎯 RECOMMENDATIONS:")
    print("1. Try opening: output/tracked_output_h264.mp4 (Most compatible)")
    print("2. If that fails, try: output/tracked_output_h265.mp4 (Smaller size)")
    print("3. For web browsers: output/tracked_output.webm")
    print("4. For older players: output/tracked_output.avi")
    print("\n📱 Video players to try:")
    print("- VLC Media Player (Free, supports everything)")
    print("- QuickTime Player (macOS default)")
    print("- Windows Media Player")
    print("- Chrome/Firefox browser (for WebM)")

if __name__ == "__main__":
    main() 
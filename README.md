# Player Re-Identification in Sports Footage (Single Feed)

A complete, practical solution for automatically **detecting, tracking, and re-identifying players** in sports match videos from a single camera feed. The system uses a YOLOv11 deep learning model for detection and the Deep SORT algorithm for robust multi-object tracking and re-identification. All core logic is implemented in a single, easy-to-use script.

---

## 🌟 Key Features

- **Player Detection:** Uses a pre-trained YOLOv11 model to identify players and the ball in each frame.
- **Player Tracking:** Employs Deep SORT, combining motion (Kalman filter) and appearance features for robust, consistent player IDs—even after occlusion.
- **Re-Identification:** Maintains player IDs when players leave and re-enter the scene.
- **Video Annotation:** Draws bounding boxes and IDs on the video for clear visualization.
- **Single-Script Pipeline:** All major functionality is in `run.py` for simplicity and portability.

---

### Installation

1.  **Create and activate the conda environment:**
    ```bash
    conda env create -f environment-cross-platform.yml
    conda activate football-player-reid
    ```

### Basic Usage

1. **Activate the environment:**
   ```bash
   conda activate football-player-reid
   ```

2. **Place your files:**
   - Put your video file in the `input/` directory as `15sec_input_720p.mp4`
   - Download the YOLOv11 model to `models/yolo_model.pt`:

3. **Run the system:**
   ```bash
   python run.py
   ```

4. **View results:**
   - The annotated video will be saved to `output/tracked_output.mp4`
   - Processing logs will be displayed in the terminal

## 🛠️ Customization

- **Change Video**: To process a different video, place it in the `input/` directory and update the `input_video_path` in `run.py`.
- **Adjust Detection Confidence**: Modify the `confidence_threshold` in `run.py` to tune the sensitivity of the player detector.
- **Tune Tracker**: The `SimpleTracker` can be tuned by adjusting the `max_disappeared` parameter in `run.py`, which controls how long a track is kept without a detection.

## 🔧 Requirements

### System Requirements
- Python 3.9 (recommended) or 3.8+
- Conda package manager ([Download here](https://docs.conda.io/en/latest/miniconda.html))
- CUDA-capable GPU (recommended) or CPU
- 8GB+ RAM recommended
- 2GB+ free disk space

### Installing Conda (if not already installed)
```bash
# Download and install Miniconda (lightweight conda installer)
# Visit: https://docs.conda.io/en/latest/miniconda.html
# Or use curl (Linux/macOS):
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Verify installation
conda --version
```

### Python Dependencies
All required packages are listed in conda environment files or `requirements.txt` (pip). Key dependencies include:
- `ultralytics==8.0.20` - YOLOv11 implementation
- `torch>=1.12.0` - PyTorch deep learning framework
- `opencv>=4.5.0` - Computer vision library
- `deep-sort-realtime>=1.2.0` - Real-time object tracking

### Environment Files
- `environment-cross-platform.yml` - **Recommended** - Works on all platforms (macOS, Linux, Windows)
- `environment.yml` - GPU-enabled version (Linux/Windows with CUDA support)
- `environment-cpu.yml` - CPU-only version (explicit CPU-only setup)
- `requirements.txt` - Pip dependencies (alternative to conda)



## 📁 Project Structure

```
football-player-reid/
├── input/
│   └── 15sec_input_720p.mp4       # Input video file
├── models/
│   └── yolo_model.pt              # YOLOv11 model file
├── output/
│   └── tracked_output.mp4         # Generated output video
├── README.md                      # This file
├── run.py                         # Main processing script
├── environment-cross-platform.yml # Conda env (recommended)
├── environment.yml                # Conda env (GPU)
├── environment-cpu.yml            # Conda env (CPU-only)
├── requirements.txt               # pip requirements
├── setup.py                       # Python package setup
├── test_detection.py              # Detection test script
├── test_video.py                  # Video processing test script
└── football_player_reid.egg-info # Packaging metadata
```

## 🔍 How It Works

### 1. Detection Phase
- **YOLOv11 Model**: Detects players (class 0) and ball (class 1) in each frame
- **Confidence Filtering**: Removes low-confidence detections
- **Area Filtering**: Filters detections by bounding box area to reduce noise

### 2. Tracking Phase
- **Deep SORT Algorithm**: Associates detections across frames using:
  - **Motion Model**: Predicts object locations based on previous positions
  - **Appearance Model**: Uses deep learning features for re-identification
  - **Data Association**: Matches detections to existing tracks

### 3. Re-Identification
- **Appearance Embeddings**: Extracts visual features for each player
- **Cosine Distance**: Compares appearance features for re-identification
- **Track Management**: Maintains tracks even when players temporarily disappear

### 4. Output Generation
- **Bounding Box Annotation**: Draws colored boxes around each player
- **ID Labels**: Displays consistent player IDs
- **Statistics Overlay**: Shows tracking statistics on video

## 📊 Performance Considerations

### Processing Speed
- **GPU**: ~15-30 FPS processing speed
- **CPU**: ~3-8 FPS processing speed
- **Memory**: ~2-4GB RAM usage during processing

### Tracking Quality
- **Optimal Conditions**: Good lighting, clear player visibility
- **Challenging Conditions**: Occlusion, rapid movement, similar uniforms
- **Re-ID Success Rate**: ~85-95% under normal conditions

## 🛠️ Troubleshooting

### Common Issues

1. **"Model file not found" Error**
   ```bash
   # Ensure the model file exists and path is correct
   ls -la models/yolo_model.pt
   ```

2. **"Video file not found" Error**
   ```bash
   # Check video file location and format
   ls -la input/15sec_input_720p.mp4
   ```

3. **CUDA Out of Memory**
   ```bash
   # Reduce batch size or use CPU
   export CUDA_VISIBLE_DEVICES=""  # Force CPU usage
   ```

4. **Import Errors**
   ```bash
   # For conda users - recreate environment (recommended)
   conda env remove -n football-player-reid
   conda env create -f environment-cross-platform.yml
   conda activate football-player-reid
   
   # For pip users - reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   
   # Test specific imports
   python -c "from src.tracker import PlayerTracker; print('✅ Tracker import OK')"
   python -c "from src.detector import PlayerDetector; print('✅ Detector import OK')"
   python -c "from src.main import PlayerReIDSystem; print('✅ Main system OK')"
   ```

5. **Conda Environment Issues**
   ```bash
   # List conda environments
   conda env list
   
   # Remove and recreate environment (cross-platform - recommended)
   conda env remove -n football-player-reid
   conda env create -f environment-cross-platform.yml
   
   # Or GPU version (Linux/Windows with CUDA)
   conda env remove -n football-player-reid
   conda env create -f environment.yml
   
   # Or CPU version
   conda env remove -n football-player-reid-cpu
   conda env create -f environment-cpu.yml
   
   # Update conda if needed
   conda update conda
   ```

6. **DeepSort Import Issues**
   ```bash
   # If you get "cannot import name 'DeepSort'" error
   # This is already fixed in the current version, but if you encounter it:
   
   # Check the correct import in your tracker.py file should be:
   # from deep_sort_realtime.deepsort_tracker import DeepSort
   
   # Test the import:
   python -c "from deep_sort_realtime.deepsort_tracker import DeepSort; print('✅ DeepSort import OK')"
   ```

7. **PyTorch CUDA Issues**
   ```bash
   # For conda users - install CUDA-enabled PyTorch
   conda activate football-player-reid
   conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
   
   # Verify CUDA installation
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

### Performance Optimization

1. **For GPU Acceleration:**
   ```bash
   # Conda users - install CUDA-enabled packages
   conda activate football-player-reid
   conda install pytorch torchvision pytorch-cuda=11.7 -c pytorch -c nvidia
   
   # Verify CUDA installation
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **For Memory Issues:**
   - Reduce video resolution
   - Process video in chunks
   - Use smaller detection area thresholds

3. **Environment Management:**
   ```bash
   # Clean conda cache to free space
   conda clean --all
   
   # Update environment with new packages (cross-platform)
   conda env update -f environment-cross-platform.yml
   
   # Or update GPU environment
   conda env update -f environment.yml
   
   # Export current environment
   conda env export > my-environment.yml
   
   # List all installed packages
   conda list
   ```

## 🔧 Configuration

### Detection Parameters
- `confidence_threshold`: Minimum detection confidence (default: 0.5)
- `min_detection_area`: Minimum bounding box area (default: 500.0)
- `max_detection_area`: Maximum bounding box area (default: 50000.0)

### Tracking Parameters
- `max_age`: Frames to keep track without detection (default: 30)
- `n_init`: Frames needed to confirm track (default: 3)
- `max_cosine_distance`: Maximum cosine distance for matching (default: 0.4)

## 📈 Expected Results

The system will produce:
- **Annotated Video**: Players with consistent colored bounding boxes
- **Player IDs**: Numeric labels that persist across frames
- **Statistics**: Real-time tracking metrics
- **Re-identification**: Same IDs for players who reappear

## 🤝 Contributing

Contributions are welcome! If you have ideas for improvements, please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📞 Support

For questions or issues:
- Review the troubleshooting section above
- Create an issue in the project repository

## 🔗 References

- [YOLOv11 Documentation](https://docs.ultralytics.com/)
- [Deep SORT Paper](https://arxiv.org/abs/1703.07402)
- [OpenCV Documentation](https://opencv.org/)
- [PyTorch Documentation](https://pytorch.org/) 

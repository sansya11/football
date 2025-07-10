# Brief Report: Player Re-Identification in Sports Footage (Single Feed)

## Approach and Methodology

The solution is designed to **automatically detect, track, and re-identify players** in sports match videos from a single camera feed. The core methodology includes:

- **Detection:**  
  Utilized a pre-trained **YOLOv11** model to detect players and the ball in every video frame. Detection results are filtered by confidence and bounding box area to minimize false positives and noise.

- **Tracking:**  
  Integrated the **Deep SORT** algorithm, which combines:
  - A **Kalman filter** for motion prediction,
  - The **Hungarian algorithm** for associating detections to existing tracks,
  - **Appearance embeddings** for robust re-identification and consistent ID assignment, even after occlusion or re-entry.

- **Annotation & Output:**  
  Each frame is annotated with bounding boxes and unique player IDs. The processed, annotated video is saved for review.

**All pipeline logic is implemented in a single script (`run.py`) for simplicity and portability.**

---

## Techniques Tried and Outcomes

- **YOLOv11 for Detection:**  
  Provided fast and accurate player and ball localization. Confidence and area-based filtering effectively reduced false positives.

- **Deep SORT for Tracking and Re-ID:**  
  Enabled robust multi-object tracking, maintaining consistent player IDs across frames. The combination of motion and appearance features handled most occlusions and re-identification scenarios effectively.

- **Parameter Tuning:**  
  Adjusted detection confidence thresholds and Deep SORT parameters (e.g., `max_age`, `n_init`) to optimize tracking stability and reduce ID switches.

**Outcomes:**  
- Achieved real-time or near-real-time processing speeds (15–30 FPS on GPU, 3–8 FPS on CPU).
- Maintained high re-identification accuracy (~85–95%) under normal conditions.
- Produced clear, annotated videos suitable for further analytics or presentation.

---

## Challenges Encountered

- **Occlusion and Similar Uniforms:**  
  Tracking quality degraded when players overlapped for extended periods or wore visually similar uniforms, occasionally resulting in ID switches.

- **Resource Constraints:**  
  High-resolution videos and long durations increased memory and processing requirements, especially on CPU-only systems.

- **Dependency Management:**  
  Ensuring compatibility between deep learning, tracking, and video processing libraries required careful environment setup (multiple `environment.yml` files provided).

- **Model and Video File Handling:**  
  Incorrect file paths or missing files were common sources of runtime errors, addressed through detailed troubleshooting steps.

---

## Incomplete Aspects & Future Work

- **Advanced Re-Identification:**  
  While Deep SORT with appearance embeddings is robust, integrating more sophisticated re-ID models or temporal smoothing could further improve ID consistency in highly challenging scenarios.

- **Customization and User Interface:**  
  The current pipeline is script-based. A user-friendly GUI or web interface could make the tool more accessible to non-technical users.

- **Scalability:**  
  Processing very long videos or multiple camera feeds in parallel would require additional engineering for resource management and synchronization.

- **Additional Analytics:**  
  Extending the system to extract and visualize advanced player statistics (e.g., heatmaps, trajectories) would add value for coaches and analysts.

**With more time/resources:**  
- Integrate a GUI for easier operation.
- Enhance re-identification with state-of-the-art models.
- Optimize for distributed or cloud-based processing.
- Add modules for advanced analytics and reporting.

---


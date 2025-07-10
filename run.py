#!/usr/bin/env python
import cv2
import numpy as np
import logging
import time
from pathlib import Path
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from filterpy.kalman import KalmanFilter
# from sklearn.utils.linear_assignment_ import linear_assignment  # Deprecated
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict, Any
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PlayerDetector:
    """Enhanced YOLO-based player detector with confidence filtering and NMS"""
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.3, nms_threshold: float = 0.5):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.model = self._load_model()
        logger.info(f"PlayerDetector initialized with confidence={confidence_threshold}, nms={nms_threshold}")
    
    def _load_model(self):
        """Load YOLO model with error handling"""
        try:
            import ultralytics
            model = ultralytics.YOLO(self.model_path)
            logger.info(f"Successfully loaded YOLO model from {self.model_path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Detect players in frame and return detections in format [x1, y1, x2, y2, confidence]
        
        Args:
            frame: Input image frame
            
        Returns:
            detections: Array of shape (N, 5) with [x1, y1, x2, y2, confidence]
        """
        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            detections = []
            frame_height, frame_width = frame.shape[:2]
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    # Filter for player class (class 2) and referee class (class 3)
                    player_mask = (boxes.cls == 2) | (boxes.cls == 3)  # Players and referees
                    if player_mask.any():
                        player_boxes = boxes.xyxy[player_mask].cpu().numpy()
                        player_confs = boxes.conf[player_mask].cpu().numpy()
                        player_classes = boxes.cls[player_mask].cpu().numpy()
                        
                        # Combine boxes and confidences
                        for box, conf, cls in zip(player_boxes, player_confs, player_classes):
                            if conf >= self.confidence_threshold:
                                x1, y1, x2, y2 = box
                                
                                # Add detection (no need for strict filtering since model is accurate)
                                detections.append([x1, y1, x2, y2, conf])
                                
                                class_name = "player" if cls == 2 else "referee"
                                logger.debug(f"Added {class_name}: {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f} conf={conf:.3f}")
            
            if len(detections) == 0:
                return np.empty((0, 5))
            
            detections = np.array(detections)
            
            # Apply NMS
            if len(detections) > 1:
                detections = self._apply_nms(detections)
            
            logger.debug(f"Detected {len(detections)} players")
            return detections
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            return np.empty((0, 5))
    
    def _is_valid_player_detection(self, x1: float, y1: float, x2: float, y2: float, 
                                 frame_width: int, frame_height: int) -> bool:
        """
        Validate if a detection is likely a player (not ball, referee, etc.)
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            frame_width, frame_height: Frame dimensions
            
        Returns:
            bool: True if detection is likely a player
        """
        # Calculate box dimensions
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Relative dimensions
        rel_width = width / frame_width
        rel_height = height / frame_height
        rel_area = area / (frame_width * frame_height)
        
        # Aspect ratio (height/width)
        aspect_ratio = height / max(width, 1)
        
        # Player validation criteria (filter out small objects like balls)
        # 1. Minimum size (filter out very small objects like balls)
        min_rel_width = 0.025  # At least 2.5% of frame width (balls are ~1%)
        min_rel_height = 0.04  # At least 4% of frame height (balls are ~1.5-2%)
        min_rel_area = 0.0008  # At least 0.08% of frame area
        
        # 2. Maximum size (filter out very large detections)
        max_rel_width = 0.4   # At most 40% of frame width
        max_rel_height = 0.9  # At most 90% of frame height
        max_rel_area = 0.25   # At most 25% of frame area
        
        # 3. Aspect ratio (players are typically taller than wide, but be flexible)
        min_aspect_ratio = 1.0  # Height should be at least equal to width
        max_aspect_ratio = 6.0  # But not too tall
        
        # 4. Position validation (players shouldn't be at very top of frame)
        min_y_position = 0.05 * frame_height  # Not in top 5% of frame
        
        # Apply filters
        size_valid = (rel_width >= min_rel_width and rel_width <= max_rel_width and
                     rel_height >= min_rel_height and rel_height <= max_rel_height and
                     rel_area >= min_rel_area and rel_area <= max_rel_area)
        
        aspect_valid = (aspect_ratio >= min_aspect_ratio and aspect_ratio <= max_aspect_ratio)
        
        position_valid = y1 >= min_y_position
        
        is_valid = size_valid and aspect_valid and position_valid
        
        if not is_valid:
            logger.debug(f"Filtered detection: size={rel_width:.3f}x{rel_height:.3f}, "
                        f"aspect={aspect_ratio:.2f}, y1={y1:.0f}")
        
        return is_valid
    
    def _apply_nms(self, detections: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return detections
        
        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return detections[keep]


import torchreid
from torchvision import transforms
from PIL import Image

class AppearanceExtractor:
    def __init__(self, feature_dim: int = 512):
        self.feature_dim = feature_dim
        logger.info("AppearanceExtractor initialized")
        # Load strong ReID model
        self.deep_model = torchreid.models.build_model(
            name='osnet_x0_25', num_classes=1000, pretrained=True
        )
        self.deep_model.eval()
        self.deep_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
        try:
            x1, y1, x2, y2 = bbox[:4].astype(int)
            h, w = frame.shape[:2]
            x1 = max(0, min(x1, w-1))
            y1 = max(0, min(y1, h-1))
            x2 = max(x1+1, min(x2, w))
            y2 = max(y1+1, min(y2, h))
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                return np.zeros(self.feature_dim + 512)  # 512 for deep features

            roi_resized = cv2.resize(roi, (64, 128))

            features = []

            # 1. Color histogram features
            hist_features = self._extract_color_histogram(roi_resized)
            features.extend(hist_features)

            # 2. Texture features
            texture_features = self._extract_texture_features(roi_resized)
            features.extend(texture_features)

            # 3. Spatial features
            spatial_features = self._extract_spatial_features(bbox, frame.shape)
            features.extend(spatial_features)

            # 4. Deep ReID features
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            deep_input = self.deep_transform(roi_pil).unsqueeze(0)
            with torch.no_grad():
                deep_feat = self.deep_model(deep_input)
            deep_feat = deep_feat.cpu().numpy().flatten()
            deep_feat = deep_feat / (np.linalg.norm(deep_feat) + 1e-6)
            features.extend(deep_feat)
            
            all_features = np.concatenate([hist_features, texture_features, spatial_features, deep_feat])

            # Pad or truncate to desired dimension
            features = np.array(features)
            total_dim = self.feature_dim + 512
            if len(features) > total_dim:
                features = features[:total_dim]
            elif len(features) < total_dim:
                features = np.pad(features, (0, total_dim - len(features)))

            # Normalize
            norm = np.linalg.norm(features)
            if norm > 0:
                features = features / norm

            return features

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return np.zeros(self.feature_dim + 512)
    
    def _extract_color_histogram(self, roi: np.ndarray, bins: int = 32) -> List[float]:
        """Extract color histogram features"""
        features = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        
        # HSV histograms
        for i in range(3):
            hist = cv2.calcHist([hsv], [i], None, [bins//4], [0, 256])
            features.extend(hist.flatten())
        
        # LAB histograms  
        for i in range(3):
            hist = cv2.calcHist([lab], [i], None, [bins//4], [0, 256])
            features.extend(hist.flatten())
        
        return features
    
    def _extract_texture_features(self, roi: np.ndarray) -> List[float]:
        """Extract texture features using gradients"""
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        # Statistical features
        features = [
            np.mean(magnitude), np.std(magnitude),
            np.mean(direction), np.std(direction),
            np.mean(gray), np.std(gray)
        ]
        
        return features
    
    def _extract_spatial_features(self, bbox: np.ndarray, frame_shape: Tuple[int, int]) -> List[float]:
        """Extract spatial and geometric features"""
        x1, y1, x2, y2 = bbox[:4]
        h, w = frame_shape[:2]
        
        # Normalize coordinates
        center_x = (x1 + x2) / (2 * w)
        center_y = (y1 + y2) / (2 * h)
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        aspect_ratio = width / max(height, 1e-6)
        area = width * height
        
        return [center_x, center_y, width, height, aspect_ratio, area]


class KalmanBoxTracker:
    """Enhanced Kalman filter for tracking bounding boxes with velocity and acceleration"""
    
    count = 0
    
    def __init__(self, bbox: np.ndarray, appearance_features: np.ndarray = None):
        """
        Initialize tracker with a bounding box
        
        Args:
            bbox: [x1, y1, x2, y2, confidence]
            appearance_features: Appearance feature vector
        """
        # Define state: [x, y, s, r, vx, vy, vs, vr] where s=scale, r=aspect_ratio
        self.kf = KalmanFilter(dim_x=8, dim_z=4)
        
        # State transition matrix (constant velocity model)
        self.kf.F = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],  # x = x + vx
            [0, 1, 0, 0, 0, 1, 0, 0],  # y = y + vy  
            [0, 0, 1, 0, 0, 0, 1, 0],  # s = s + vs
            [0, 0, 0, 1, 0, 0, 0, 1],  # r = r + vr
            [0, 0, 0, 0, 1, 0, 0, 0],  # vx = vx
            [0, 0, 0, 0, 0, 1, 0, 0],  # vy = vy
            [0, 0, 0, 0, 0, 0, 1, 0],  # vs = vs
            [0, 0, 0, 0, 0, 0, 0, 1]   # vr = vr
        ])
        
        # Measurement function (we observe x, y, s, r)
        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ])
        
        # Measurement noise - tuned for football players
        self.kf.R[2:, 2:] *= 5.0     # Reduced noise for scale/aspect ratio
        
        # Process noise - tuned for football player movement
        self.kf.P[4:, 4:] *= 1000.0  # High uncertainty for velocities
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.005   # Very low process noise for aspect ratio velocity
        self.kf.Q[4:, 4:] *= 0.02    # Moderate process noise for player movement
        
        # Initialize state
        self.kf.x[:4] = self._convert_bbox_to_z(bbox)
        
        # Tracking state
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        
        # Appearance features
        self.appearance_features = []
        if appearance_features is not None:
            self.appearance_features.append(appearance_features)
        
        # Enhanced tracking state
        self.confidence_history = [bbox[4] if len(bbox) > 4 else 1.0]
        self.lost_count = 0
        self.max_lost_frames = 30
        
        logger.debug(f"Created tracker {self.id}")
    
    def update(self, bbox: np.ndarray, appearance_features: np.ndarray = None):
        """Update tracker with new detection"""
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.lost_count = 0
        
        # Update Kalman filter
        self.kf.update(self._convert_bbox_to_z(bbox))
        
        # Update appearance features
        if appearance_features is not None:
            self.appearance_features.append(appearance_features)
            # Keep only recent features (sliding window)
            if len(self.appearance_features) > 10:
                self.appearance_features.pop(0)
        
        # Update confidence history
        if len(bbox) > 4:
            self.confidence_history.append(bbox[4])
            if len(self.confidence_history) > 10:
                self.confidence_history.pop(0)
        
        logger.debug(f"Updated tracker {self.id}")
    
    def predict(self):
        """Predict next state"""
        # Handle aspect ratio constraints
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        
        self.kf.predict()
        self.age += 1
        
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.lost_count += 1
        
        self.time_since_update += 1
        self.history.append(self._convert_x_to_bbox(self.kf.x))
        
        return self.history[-1]
    
    def get_state(self) -> np.ndarray:
        """Return current bounding box estimate"""
        return self._convert_x_to_bbox(self.kf.x)
    
    def get_average_appearance_feature(self) -> np.ndarray:
        """Get average appearance feature over recent observations"""
        if not self.appearance_features:
            return np.array([])
        
        features = np.array(self.appearance_features)
        return np.mean(features, axis=0)
    
    def get_confidence(self) -> float:
        """Get average confidence over recent observations"""
        if not self.confidence_history:
            return 0.5
        return np.mean(self.confidence_history)
    
    def is_tentative(self) -> bool:
        """Check if tracker is still tentative (needs more hits)"""
        return self.hit_streak < 2  # Reduced from 3 to 2
    
    def is_confirmed(self) -> bool:
        """Check if tracker is confirmed"""
        return self.hit_streak >= 2  # Reduced from 3 to 2
    
    def is_lost(self) -> bool:
        """Check if tracker should be considered lost"""
        return self.lost_count > self.max_lost_frames
    
    def _convert_bbox_to_z(self, bbox: np.ndarray) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [x, y, s, r] format"""
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = bbox[0] + w / 2.0
        y = bbox[1] + h / 2.0
        s = w * h  # scale (area)
        r = w / max(h, 1e-6)  # aspect ratio
        return np.array([x, y, s, r]).reshape((4, 1))
    
    def _convert_x_to_bbox(self, x: np.ndarray, score: float = None) -> np.ndarray:
        """Convert [x, y, s, r] to [x1, y1, x2, y2] format"""
        w = np.sqrt(x[2] * x[3])
        h = x[2] / max(w, 1e-6)
        
        x1 = x[0] - w / 2.0
        y1 = x[1] - h / 2.0
        x2 = x[0] + w / 2.0
        y2 = x[1] + h / 2.0
        
        if score is None:
            return np.array([x1, y1, x2, y2]).flatten()
        else:
            return np.array([x1, y1, x2, y2, score]).flatten()


class ByteTracker:
    """
    Enhanced ByteTrack implementation with appearance features and improved association
    Based on: https://arxiv.org/abs/2110.06864
    """
    
    def __init__(self, frame_rate: int = 30, track_thresh: float = 0.5, track_buffer: int = 120,
                 match_thresh: float = 0.8, high_thresh: float = 0.6, low_thresh: float = 0.1,
                 appearance_thresh: float = 0.15):
        """
        Initialize ByteTracker
        
        Args:
            frame_rate: Video frame rate
            track_thresh: Detection confidence threshold for tracking
            track_buffer: Number of frames to keep lost tracks
            match_thresh: Matching threshold for first association
            high_thresh: High confidence detection threshold  
            low_thresh: Low confidence detection threshold
            appearance_thresh: Appearance similarity threshold
        """
        self.frame_rate = frame_rate
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self.high_thresh = high_thresh
        self.low_thresh = low_thresh
        self.appearance_thresh = appearance_thresh
        
        # Tracker lists
        self.tracked_stracks = []  # Confirmed tracks
        self.lost_stracks = []     # Lost tracks
        self.removed_stracks = []  # Removed tracks
        
        # Frame counter
        self.frame_id = 0
        
        # Appearance extractor
        self.appearance_extractor = AppearanceExtractor()
        
        logger.info(f"ByteTracker initialized with track_thresh={track_thresh}, "
                   f"high_thresh={high_thresh}, low_thresh={low_thresh}")
    
    def update(self, frame: np.ndarray, detections: np.ndarray) -> List[np.ndarray]:
        """
        Update tracker with new frame and detections
        
        Args:
            frame: Current frame
            detections: Array of detections [x1, y1, x2, y2, confidence]
            
        Returns:
            List of active tracks [x1, y1, x2, y2, track_id]
        """
        self.frame_id += 1
        
        # Extract appearance features for all detections
        det_features = []
        for det in detections:
            features = self.appearance_extractor.extract_features(frame, det)
            det_features.append(features)
        det_features = np.array(det_features) if det_features else np.empty((0, 512))
        
        # Predict existing tracks
        for track in self.tracked_stracks:
            track.predict()
        for track in self.lost_stracks:
            track.predict()
        
        # Separate high and low confidence detections
        if len(detections) > 0:
            high_conf_mask = detections[:, 4] >= self.high_thresh
            low_conf_mask = (detections[:, 4] >= self.low_thresh) & (detections[:, 4] < self.high_thresh)
            
            high_conf_dets = detections[high_conf_mask]
            low_conf_dets = detections[low_conf_mask]
            
            high_conf_features = det_features[high_conf_mask] if len(det_features) > 0 else np.empty((0, 512))
            low_conf_features = det_features[low_conf_mask] if len(det_features) > 0 else np.empty((0, 512))
        else:
            high_conf_dets = np.empty((0, 5))
            low_conf_dets = np.empty((0, 5))
            high_conf_features = np.empty((0, 512))
            low_conf_features = np.empty((0, 512))
        
        # First association: high confidence detections with tracked tracks
        track_pool = self.tracked_stracks
        tracked_tracks, unmatched_dets, unmatched_tracks = self._associate_tracks_dets(
            track_pool, high_conf_dets, high_conf_features, self.match_thresh
        )
        
        # Update matched tracks
        for track_idx, det_idx in tracked_tracks:
            track = track_pool[track_idx]
            det = high_conf_dets[det_idx]
            features = high_conf_features[det_idx] if len(high_conf_features) > 0 else None
            track.update(det, features)
        
        # Second association: unmatched tracks with low confidence detections
        unmatched_track_pool = [track_pool[i] for i in unmatched_tracks]
        second_tracked, second_unmatched_dets, second_unmatched_tracks = self._associate_tracks_dets(
            unmatched_track_pool, low_conf_dets, low_conf_features, 0.5
        )
        
        # Update second round matches
        for track_idx, det_idx in second_tracked:
            track = unmatched_track_pool[track_idx]
            det = low_conf_dets[det_idx]
            features = low_conf_features[det_idx] if len(low_conf_features) > 0 else None
            track.update(det, features)
        
        # Mark unmatched tracks as lost
        for track_idx in second_unmatched_tracks:
            track = unmatched_track_pool[track_idx]
            track.time_since_update += 1
        
        # Third association: lost tracks with unmatched high confidence detections
        unmatched_high_dets = high_conf_dets[[i for i in unmatched_dets if i not in [pair[1] for pair in second_tracked]]]
        unmatched_high_features = high_conf_features[[i for i in unmatched_dets if i not in [pair[1] for pair in second_tracked]]] if len(high_conf_features) > 0 else np.empty((0, 512))
        
        lost_tracked, lost_unmatched_dets, lost_unmatched_tracks = self._associate_tracks_dets(
            self.lost_stracks, unmatched_high_dets, unmatched_high_features, 0.4
        )
        
        # Reactivate matched lost tracks
        for track_idx, det_idx in lost_tracked:
            track = self.lost_stracks[track_idx]
            det = unmatched_high_dets[det_idx]
            features = unmatched_high_features[det_idx] if len(unmatched_high_features) > 0 else None
            track.update(det, features)
            self.tracked_stracks.append(track)
        
        # Remove reactivated tracks from lost list
        for track_idx in sorted([pair[0] for pair in lost_tracked], reverse=True):
            self.lost_stracks.pop(track_idx)
        
        # Create new tracks for remaining unmatched detections
        remaining_dets = unmatched_high_dets[[i for i in lost_unmatched_dets]]
        remaining_features = unmatched_high_features[[i for i in lost_unmatched_dets]] if len(unmatched_high_features) > 0 else []
        
        for i, det in enumerate(remaining_dets):
            features = remaining_features[i] if len(remaining_features) > i else None
            new_track = KalmanBoxTracker(det, features)
            self.tracked_stracks.append(new_track)
        
        # Move lost tracks to lost list
        lost_tracks = []
        for track in self.tracked_stracks:
            if track.time_since_update > 1:
                lost_tracks.append(track)
        
        for track in lost_tracks:
            self.tracked_stracks.remove(track)
            self.lost_stracks.append(track)
        
        # Remove old lost tracks
        self.lost_stracks = [track for track in self.lost_stracks 
                           if track.time_since_update <= self.track_buffer]
        
        # Get output tracks
        output_tracks = []
        for track in self.tracked_stracks:
            if track.is_confirmed() and track.time_since_update <= 1:
                bbox = track.get_state()
                output_tracks.append(np.array([bbox[0], bbox[1], bbox[2], bbox[3], track.id]))
        
        logger.debug(f"Frame {self.frame_id}: {len(output_tracks)} active tracks, "
                    f"{len(self.lost_stracks)} lost tracks")
        
        # Debug logging for tracking issues
        if self.frame_id % 30 == 0:
            logger.info(f"Frame {self.frame_id}: {len(detections)} detections, "
                       f"{len(high_conf_dets)} high conf, {len(low_conf_dets)} low conf")
            logger.info(f"Tracked: {len(self.tracked_stracks)}, Lost: {len(self.lost_stracks)}, "
                       f"Output: {len(output_tracks)}")
        
        return output_tracks
    
    def _associate_tracks_dets(self, tracks: List[KalmanBoxTracker], detections: np.ndarray, 
                              features: np.ndarray, thresh: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Associate tracks with detections using IoU and appearance features"""
        if len(tracks) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(tracks)))
        
        # Get track predictions
        track_boxes = []
        track_features = []
        
        for track in tracks:
            track_boxes.append(track.get_state())
            avg_features = track.get_average_appearance_feature()
            track_features.append(avg_features if len(avg_features) > 0 else np.zeros(512))
        
        track_boxes = np.array(track_boxes)
        track_features = np.array(track_features)
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(track_boxes, detections[:, :4])
        
        # Compute appearance similarity matrix
        if len(features) > 0 and len(track_features) > 0:
            # Normalize features to unit vectors for cosine similarity
            track_norms = np.linalg.norm(track_features, axis=1, keepdims=True)
            det_norms = np.linalg.norm(features, axis=1, keepdims=True)
            
            # Avoid division by zero
            track_norms = np.where(track_norms == 0, 1, track_norms)
            det_norms = np.where(det_norms == 0, 1, det_norms)
            
            track_features_norm = track_features / track_norms
            features_norm = features / det_norms
            
            # Cosine similarity
            app_matrix = np.dot(track_features_norm, features_norm.T)
            # Clip to [-1, 1] range and normalize to [0, 1]
            app_matrix = np.clip(app_matrix, -1, 1)
            app_matrix = (app_matrix + 1) / 2
        else:
            app_matrix = np.zeros((len(tracks), len(detections)))
        
        # Combined cost matrix (IoU + appearance)
        cost_matrix = -(0.7 * iou_matrix + 0.3 * app_matrix)
        
        # Ensure numerical stability
        cost_matrix = np.nan_to_num(cost_matrix, nan=1.0, posinf=1.0, neginf=-1.0)
        
        # Hungarian assignment
        if cost_matrix.size > 0 and cost_matrix.shape[0] > 0 and cost_matrix.shape[1] > 0:
            try:
                matched_indices = linear_sum_assignment(cost_matrix)
                matched_pairs = list(zip(matched_indices[0], matched_indices[1]))
            except ValueError as e:
                logger.warning(f"Assignment failed: {e}, skipping this association")
                matched_pairs = []
        else:
            matched_pairs = []
        
        # Filter matches by threshold
        valid_matches = []
        for track_idx, det_idx in matched_pairs:
            if (-cost_matrix[track_idx, det_idx]) >= thresh:
                valid_matches.append((track_idx, det_idx))
        
        # Get unmatched tracks and detections
        matched_track_indices = [pair[0] for pair in valid_matches]
        matched_det_indices = [pair[1] for pair in valid_matches]
        
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_dets = [i for i in range(len(detections)) if i not in matched_det_indices]
        
        return valid_matches, unmatched_dets, unmatched_tracks
    
    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))
        
        # Expand dimensions for broadcasting
        boxes1 = np.expand_dims(boxes1, axis=1)  # (N, 1, 4)
        boxes2 = np.expand_dims(boxes2, axis=0)  # (1, M, 4)
        
        # Compute intersection
        x1 = np.maximum(boxes1[..., 0], boxes2[..., 0])
        y1 = np.maximum(boxes1[..., 1], boxes2[..., 1])
        x2 = np.minimum(boxes1[..., 2], boxes2[..., 2])
        y2 = np.minimum(boxes1[..., 3], boxes2[..., 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute areas
        area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        
        # Compute IoU
        union = area1 + area2 - intersection
        iou = intersection / np.maximum(union, 1e-6)
        
        return iou


class VideoHandler:
    """Enhanced video processing with frame skipping and optimization"""
    
    def __init__(self, input_path: str, output_path: str, skip_frames: int = 0):
        self.input_path = input_path
        self.output_path = output_path
        self.skip_frames = skip_frames
        self.cap = None
        self.writer = None
        self.total_frames = 0
        self.processed_frames = 0
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.input_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.input_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize writer with H.264 codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        # Fallback to mp4v if H.264 fails
        if not self.writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, (self.width, self.height))
        
        logger.info(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()
    
    def read_frame(self):
        """Read next frame, applying frame skipping if configured"""
        if self.skip_frames > 0:
            for _ in range(self.skip_frames):
                ret, _ = self.cap.read()
                if not ret:
                    return False, None
        
        ret, frame = self.cap.read()
        if ret:
            self.processed_frames += 1
        return ret, frame
    
    def write_frame(self, frame):
        """Write frame to output video"""
        if self.writer:
            self.writer.write(frame)
    
    def get_progress(self):
        """Get processing progress as percentage"""
        if self.total_frames == 0:
            return 0
        return (self.processed_frames / self.total_frames) * 100


def draw_tracks(frame: np.ndarray, tracks: List[np.ndarray], colors: dict = None) -> np.ndarray:
    """
    Draw tracking results on frame with enhanced visualization
    
    Args:
        frame: Input frame
        tracks: List of tracks [x1, y1, x2, y2, track_id]
        colors: Dictionary mapping track_id to color
        
    Returns:
        Annotated frame
    """
    if colors is None:
        colors = {}
    
    annotated_frame = frame.copy()
    
    for track in tracks:
        if len(track) < 5:
            continue
            
        x1, y1, x2, y2, track_id = track[:5]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = int(track_id)
        
        # Generate consistent color for track
        if track_id not in colors:
            np.random.seed(track_id)
            colors[track_id] = (
                np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255)
            )
        
        color = colors[track_id]
        
        # Draw bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw track ID with background
        label = f"ID: {track_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        
        # Background rectangle for text
        cv2.rectangle(annotated_frame, 
                     (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0] + 10, y1), 
                     color, -1)
        
        # Text
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(annotated_frame, (center_x, center_y), 3, color, -1)
    
    # Add tracking statistics
    stats_text = f"Active Tracks: {len(tracks)}"
    cv2.putText(annotated_frame, stats_text, (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_frame


def main():
    """Main tracking pipeline with enhanced error handling and logging"""
    # Configuration
    model_path = "models/yolo_model.pt"
    input_video_path = "input/15sec_input_720p.mp4"
    output_video_path = "output/tracked_output.mp4"
    
    try:
        # 1. Initialize system components
        logger.info("Initializing system components...")
        detector = PlayerDetector(model_path, confidence_threshold=0.2)
        tracker = ByteTracker(
            frame_rate=25,
            track_thresh=0.3,      # Increased for more stable tracks
            track_buffer=60,       # Increased buffer for better re-identification
            match_thresh=0.5,      # More lenient matching
            high_thresh=0.4,       # Balanced threshold for players
            low_thresh=0.15,       # Allow lower confidence for continuity
            appearance_thresh=0.3  # Balanced appearance matching
        )
        logger.info("Components initialized successfully")
        
        # 2. Process video
        logger.info("Starting video processing...")
        colors = {}
        frame_count = 0
        start_time = time.time()
        
        with VideoHandler(input_video_path, output_video_path) as video_handler:
            while True:
                ret, frame = video_handler.read_frame()
                if not ret:
                    break
                
                frame_count += 1
                
                # 3. Detect players
                detections = detector.detect(frame)
                
                # Debug: Log detection info
                if frame_count % 30 == 0:
                    logger.info(f"Frame {frame_count}: {len(detections)} detections")
                    if len(detections) > 0:
                        logger.info(f"Detection confidences: {detections[:, 4]}")
                        # Log detection sizes to help identify players vs other objects
                        for i, det in enumerate(detections):
                            x1, y1, x2, y2 = det[:4]
                            width = x2 - x1
                            height = y2 - y1
                            rel_width = width / frame.shape[1]
                            rel_height = height / frame.shape[0]
                            aspect_ratio = height / max(width, 1)
                            logger.info(f"  Detection {i}: size={rel_width:.3f}x{rel_height:.3f}, aspect={aspect_ratio:.2f}")
                
                # 4. Update tracker
                tracks = tracker.update(frame, detections)
                
                # 5. Draw results
                annotated_frame = draw_tracks(frame, tracks, colors)
                
                # 6. Write output frame
                video_handler.write_frame(annotated_frame)
                
                # 7. Progress logging
                if frame_count % 30 == 0:  # Log every 30 frames
                    progress = video_handler.get_progress()
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    
                    logger.info(f"Progress: {progress:.1f}% | Frame: {frame_count} | "
                              f"FPS: {fps:.1f} | Active tracks: {len(tracks)}")
        
        # 8. Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        
        logger.info("="*50)
        logger.info("TRACKING COMPLETED SUCCESSFULLY")
        logger.info("="*50)
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"Average FPS: {avg_fps:.2f}")
        logger.info(f"Input video: {input_video_path}")
        logger.info(f"Output video: {output_video_path}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error(f"Error in main pipeline: {e}")
        raise


if __name__ == "__main__":
    main() 
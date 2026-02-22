import cv2
import numpy as np
from PIL import Image
from backend.utils.logging import logger
import tempfile
import os

def sample_frames(video_bytes: bytes, num_frames: int = 12) -> list[Image.Image]:
    """Extracts a fixed number of evenly spaced frames from a video buffer."""
    
    
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(video_bytes)
            
        logger.info(f"Sampling {num_frames} frames from video")
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file to extract frames")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 300 
            
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for target_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
            ret, frame = cap.read()
            if not ret: 
                break
            
            
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
            
        cap.release()
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return frames

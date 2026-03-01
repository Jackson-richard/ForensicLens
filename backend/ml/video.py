import cv2
import numpy as np
from PIL import Image
from backend.utils.logging import logger
import tempfile
import os

def sample_frames(video_bytes: bytes) -> list[Image.Image]:
    """Extracts 1 frame per second from a video buffer."""
    
    temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
    try:
        with os.fdopen(temp_fd, 'wb') as f:
            f.write(video_bytes)
            
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file to extract frames")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            fps = 30 # fallback
        if total_frames <= 0:
            total_frames = 300 
            
        duration_seconds = max(1, int(total_frames / fps))
        logger.info(f"Sampling {duration_seconds} frames (1 fps) from video")
        
        frames = []
        for sec in range(duration_seconds):
            target_idx = int(sec * fps)
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

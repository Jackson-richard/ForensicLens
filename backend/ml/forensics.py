import numpy as np
from PIL import Image

def check_exif(image: Image.Image) -> bool:
    """Check if standard EXIF metadata is present."""
    try:
        exif = image.getexif()
        return bool(exif)
    except Exception:
        return False

def check_compression_artifacts(image: Image.Image) -> bool:
    """
    Apply FFT on grayscale image and detect abnormal high-frequency spikes.
    """
    try:
        import cv2
        img_gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
        
        h, w = magnitude_spectrum.shape
        cy, cx = h // 2, w // 2
        
        # Mask out the low frequencies (center 25% radius)
        radius = min(h, w) // 4
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        mask = x*x + y*y <= radius*radius
        
        high_freq_spectrum = magnitude_spectrum.copy()
        high_freq_spectrum[mask] = 0
        
        mean_val = np.mean(magnitude_spectrum)
        max_hf = np.max(high_freq_spectrum)
        
        # Flag abnormal spikes: e.g., if max spike in high freq is > 3 * mean magnitude
        return bool(max_hf > 3.0 * mean_val)
    except Exception:
        return False

def check_noise_inconsistency(image: Image.Image) -> bool:
    """
    Estimate noise variance using a Laplacian filter over grid patches.
    """
    try:
        import cv2
        img_gray = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
        
        laplacian = cv2.Laplacian(img_gray, cv2.CV_64F)
        
        h, w = laplacian.shape
        ph, pw = h // 4, w // 4
        
        variances = []
        for i in range(4):
            for j in range(4):
                patch = laplacian[i*ph:(i+1)*ph, j*pw:(j+1)*pw]
                if patch.size > 0:
                    variances.append(np.var(patch))
                    
        if not variances:
            return False
            
        min_var = np.min(variances)
        max_var = np.max(variances)
        
        if min_var == 0:
            return bool(max_var > 0)
            
        return bool((max_var / min_var) > 10.0)
    except Exception:
        return False

def check_face_distortion(image: Image.Image) -> bool:
    """
    Use MediaPipe landmarks to check symmetry distortion.
    """
    try:
        import mediapipe as mp
        mp_face_mesh = mp.solutions.face_mesh
        img_arr = np.array(image.convert("RGB"))
        
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5) as face_mesh:
            
            results = face_mesh.process(img_arr)
            if not results.multi_face_landmarks:
                return False 
                
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Known symmetrical indices
            # Left cheek: 234, Right cheek: 454
            # Nose tip: 1
            left_cheek = np.array([landmarks[234].x, landmarks[234].y])
            right_cheek = np.array([landmarks[454].x, landmarks[454].y])
            nose = np.array([landmarks[1].x, landmarks[1].y])
            
            dist_left = np.linalg.norm(left_cheek - nose)
            dist_right = np.linalg.norm(right_cheek - nose)
            
            if max(dist_left, dist_right) == 0:
                return False
                
            ratio = min(dist_left, dist_right) / max(dist_left, dist_right)
            
            # Flag if ratio is less than 0.8 (20% asymmetry)
            return bool(ratio < 0.8)
    except Exception:
        return False

def analyze_forensics(image: Image.Image) -> dict:
    """
    Execute all forensic checks and return combined JSON-friendly dict.
    """
    return {
        "metadata_present": check_exif(image),
        "high_frequency_anomaly": check_compression_artifacts(image),
        "noise_inconsistency": check_noise_inconsistency(image),
        "face_distortion_detected": check_face_distortion(image)
    }

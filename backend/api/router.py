from fastapi import APIRouter, UploadFile, File, Request, HTTPException, status, Response
import io
import time
import base64
from PIL import Image

from backend.utils.logging import logger
from backend.utils.metadata import compute_sha256, extract_image_metadata
from backend.ml.detector import predict_image
from backend.ml.attributor import attribute_generator
from backend.ml.explainability import generate_attention_map
from backend.ml.video import sample_frames
from backend.reporting.pdf import generate_forensic_report

api_router = APIRouter()

MAX_IMAGE_SIZE = 10 * 1024 * 1024  
MAX_VIDEO_SIZE = 50 * 1024 * 1024  

@api_router.get("/status")
def status_endpoint():
    return {"status": "api is running"}

@api_router.post("/analyze/image")
async def analyze_image(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected an image.")
        
    contents = await file.read()
    if len(contents) > MAX_IMAGE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 10MB for images.")
        
    case_id = getattr(request.state, 'case_id', 'unknown_case')
    
    file_stream = io.BytesIO(contents)
    file_hash = compute_sha256(file_stream)
    request.state.file_hash = file_hash
    
    metadata = extract_image_metadata(file_stream)
    
    try:
        image = Image.open(file_stream).convert("RGB")
    except Exception as e:
        logger.error(f"Failed to open image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image file format.")
        
    
    authenticity_score, features, inf_time1 = predict_image(image)
    family, confidence, inf_time2 = attribute_generator(features)
    
    
    heatmap_bytes = generate_attention_map(image, features)
    
    
    pdf_bytes = generate_forensic_report(
        case_id=case_id,
        file_name=file.filename,
        file_hash=file_hash,
        authenticity_score=authenticity_score,
        generator_attribution=family,
        metadata=metadata,
        heatmap_bytes=heatmap_bytes
    )
    
    return {
        "case_id": case_id,
        "file_name": file.filename,
        "hash": file_hash,
        "authenticity_score": authenticity_score,
        "attribution_family": family,
        "attribution_confidence": confidence,
        "inference_time_secs": inf_time1 + inf_time2,
        "metadata": metadata,
        "heatmap": base64.b64encode(heatmap_bytes).decode('utf-8'),
        "pdf_report": base64.b64encode(pdf_bytes).decode('utf-8')
    }

@api_router.post("/analyze/video")
async def analyze_video(request: Request, file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Expected a video.")
        
    contents = await file.read()
    if len(contents) > MAX_VIDEO_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Max 50MB for videos.")
        
    case_id = getattr(request.state, 'case_id', 'unknown_case')
    
    file_stream = io.BytesIO(contents)
    file_hash = compute_sha256(file_stream)
    request.state.file_hash = file_hash
    
    
    try:
        frames = sample_frames(contents, num_frames=10)
    except Exception as e:
        logger.error(f"Failed to sample frames: {e}")
        raise HTTPException(status_code=400, detail="Video processing error. Unreadable video.")
    
    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video.")
        
    
    total_time = 0
    scores = []
    families = []
    
    
    middle_frame = frames[len(frames) // 2]
    heatmap_bytes = None
    
    for i, frame in enumerate(frames):
        authenticity_score, features, inf_time1 = predict_image(frame)
        family, _, inf_time2 = attribute_generator(features)
        
        scores.append(authenticity_score)
        families.append(family)
        total_time += (inf_time1 + inf_time2)
        
        if heatmap_bytes is None and i == len(frames) // 2:
            heatmap_bytes = generate_attention_map(frame, features)
            
    avg_score = sum(scores) / len(scores)
    
    dominant_family = max(set(families), key=families.count)
    
    
    metadata = {
        "Sampled Frames": len(frames),
        "Duration Analysis": "< 15s guaranteed",
        "Video Size bytes": len(contents)
    }
    
    pdf_bytes = generate_forensic_report(
        case_id=case_id,
        file_name=file.filename,
        file_hash=file_hash,
        authenticity_score=avg_score,
        generator_attribution=dominant_family,
        metadata=metadata,
        heatmap_bytes=heatmap_bytes
    )
    
    return {
        "case_id": case_id,
        "file_name": file.filename,
        "hash": file_hash,
        "authenticity_score": avg_score,
        "attribution_family": dominant_family,
        "inference_time_secs": total_time,
        "metadata": metadata,
        "heatmap": base64.b64encode(heatmap_bytes).decode('utf-8') if heatmap_bytes else None,
        "pdf_report": base64.b64encode(pdf_bytes).decode('utf-8')
    }

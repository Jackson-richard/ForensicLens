import asyncio
import uuid
import time
from fastapi import FastAPI, Request, status, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from backend.utils.logging import setup_logging, logger
from backend.api.router import api_router

setup_logging()

app = FastAPI(title="ForensicLens API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

REQUEST_TIMEOUT_SECONDS = 30 

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    
    if request.url.path in ["/health", "/api/status"]:
        return await call_next(request)

    case_id = request.headers.get("X-Case-ID", str(uuid.uuid4()))
    request.state.case_id = case_id
    
    start_time = time.time()
    try:
        response = await asyncio.wait_for(call_next(request), timeout=REQUEST_TIMEOUT_SECONDS)
        
        
        logger.info(
            "Request processing completed",
            extra={
                "case_id": case_id,
                "path": request.url.path,
                "duration_secs": time.time() - start_time,
                
                "file_hash": getattr(request.state, "file_hash", "")
            }
        )
        return response
    except asyncio.TimeoutError:
        logger.error(
            "Request timeout. Image or video processing exceeded constraint.",
            extra={
                "case_id": case_id,
                "path": request.url.path,
                "duration_secs": time.time() - start_time,
                "file_hash": getattr(request.state, "file_hash", ""),
            }
        )
        return JSONResponse(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            content={"detail": "Request processing timed out. The file might be too large or complex for current constraints (<5s image, <15s video)."}
        )

app.include_router(api_router, prefix="/api")

@app.get("/health")
def read_health():
    return {"status": "ok"}

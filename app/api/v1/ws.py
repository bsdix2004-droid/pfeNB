""" 
app/api/v1/ws.py - WebSocket endpoints
WS /ws/jobs/{job_id}?token=xxx
"""

import uuid

import asyncio
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from jose import JWTError
from sqlalchemy import select

from app.core.security import decode_access_token
from app.db.session import AsyncSessionLocal
from app.models.job import ExtractionJob
from app.models.user import User
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(None).bind(stage="ws")

#_______ Progress map _________

PROGRESS_MAP = {
    "queued":        {"status": "Waiting...",     "progress": 0},
    "ocr_running":   {"status": "OCR Processing", "progress": 25},
    "ai_running":    {"status": "AI Analysis",           "progress": 75},
    "done":          {"status": "Processing Completed!", "progress": 100},
    "failed":        {"status": "Processing Failed",     "progress": 0},
}

#________ WebSocket endpoint _________
@router.websocket("/jobs/{job_id}")
async def job_status_ws(
    websocket: WebSocket,
    job_id: uuid.UUID,
    token: str = Query(...),
):
    # verify token JWT 
    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if user_id is None:  
            await websocket.close(code=4001, reason="Invalid token")
            return
    except JWTError:
        #if the token is invalid or expired, close the connection with an JWTError
        await websocket.close(code=4001, reason="Invalid token")
        return
    
    # Accept the WebSocket connection
    await websocket.accept()
    logger.info("WebSocket connection accepted for job: %s", job_id) 
    
    #verify if the job belongs to the user and exists in DB
    async with AsyncSessionLocal() as db:
        job_result = await db.execute(
            select(ExtractionJob).where(ExtractionJob.id == job_id)
        )
        job = job_result.scalar_one_or_none()
        
        if job is None:
            await websocket.send_json({
                "status": "Error",
                "progress": 0,
                "message": "Job not found"
            })
            await websocket.close(code=4004, reason="Job not found")
            return
        
        if str(job.triggered_by) != user_id:
            await websocket.send_json({
                "status": "Error",
                "progress": 0,
                "message": "You are not allowed to access this job"
            })
            await websocket.close(code=4003, reason="Forbidden")
            return
        
    #Send the status in real time
    try:
        last_status = None
        while True:
            async with AsyncSessionLocal() as db:
                
                # Retrieve the current job status
                job_result = await db.execute(
                    select(ExtractionJob).where(ExtractionJob.id == job_id)
                )
                job = job_result.scalar_one_or_none()
                if job is None:
                    break
                    
                current_status = job.status
                # Send only if the status has changed
                if current_status != last_status:
                    progress_info = PROGRESS_MAP.get(current_status, {
                        "status": current_status,
                        "progress": 0
                    })
                    await websocket.send_json({
                        "status": progress_info["status"],
                        "progress": progress_info["progress"],
                        "job_id": str(job_id),
                    })
                    logger.info(
                        "WebSocket sent status %s for job %s",
                        current_status,job_id
                    )
                        
                    last_status = current_status
                
                # Close the WebSocket if the job is completed  
                if current_status in ("done", "failed"):
                    logger.info(
                        "WebSocket closing for job %s -> status: %s",
                        job_id, current_status
                    )
                    await websocket.close()
                    return
                
            await asyncio.sleep(1)  # Poll every 1 second
        
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected for job: %s", job_id)
    
    
    
    
    
    
    

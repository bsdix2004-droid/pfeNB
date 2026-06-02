"""
app/workers/celery_app.py - Celery configuration
Celery is a background task system that allows running tasks asynchronously
without blocking the user interface
"""

from celery import Celery
from app.config import get_settings

settings = get_settings()

#______ create Celery instance ______
celery_app = Celery("scanalyze") #the name of the Celery application

#______ Configuration ________
celery_app.conf.update(
    # broker_url = Redis address
    # Redis is the "messenger" that transports tasks between FastAPI and Celery
    # FastAPI sends the task → Redis stores it → Celery retrieves it
    broker_url=settings.CELERY_BROKER_URL,
    
    # result_backend = where Celery stores task results
    # after the pipeline finishes → Celery stores the result in Redis
    result_backend=settings.CELERY_RESULT_BACKEND,
    
    # include = list of files containing Celery tasks
    # Celery loads the tasks from this file at startup
    include=["app.workers.task"], 
    
    # task_serializer = the format used to serialize tasks
    task_serializer="json", 
    
    result_serializer="json",
    
    # accept_content = the formats accepted by Celery
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    
    # task_track_started = tracks the "started" status of tasks; when True, Celery updates the task status as soon as it begins execution
    task_track_started=True,
    # task_acks_late = Celery confirms the task only after it finishes; if True and Celery crashes during execution, the task will be retried
    task_acks_late=True,
    
    worker_prefetch_multiplier=1, # Celery fetches 1 task at a time to ensure fair distribution among workers
)


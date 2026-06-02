# c'est un gestionnaire de connexion à la BD
"""
app/db/session.py - Database connection manager
Provides async sessions for FastAPI and sync sessions for Celery
"""
from collections.abc import AsyncGenerator
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import get_settings

settings = get_settings()

#___ Async engine (FastAPI) ______
engine = create_async_engine(
    settings.DATABASE_URL,      # @ of BD
    echo=settings.DEBUG,          # log SQL only in dev (display SQL queries in debug mode)
    pool_pre_ping=True,            # drop stale connections
    pool_size=10,
    max_overflow=20,                
    pool_recycle=3600,              #refresh the connections every hour
)
#async = non-blocking
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base — all models inherit from this."""
    pass

#Open a database session for each request and automatically close it afterward.
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency — yields a DB session per request."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
            
#____ Sync engine (Celery) ______
sync_database_url = settings.DATABASE_URL.replace(
    "postgresql+asyncpg", "postgresql+psycopg2"
)
sync_engine = create_engine(
    sync_database_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=3600,
) 

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    class_=Session,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

@contextmanager
def get_sync_db():
    """ 
    Synchronous DB session for Celery
    Used in workers/task.py because Celery does not support async.
    """  
    db = SyncSessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()        
            
            
            
            

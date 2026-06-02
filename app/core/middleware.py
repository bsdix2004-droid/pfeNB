import time

import redis.asyncio as aioredis
from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.utils.logging import get_logger

settings = get_settings()
logger = get_logger(None).bind(stage="http")

_redis: aioredis.Redis | None = None


def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    return _redis


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=duration_ms,
        )
        # Add timing header in non-prod
        if not settings.is_prod:
            response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter backed by Redis.
    Key: IP address (or user ID if authenticated).
    Limit: `limit` requests per 60-second window.
    """

    def __init__(self, app, limit: int = 60):
        super().__init__(app)
        self.limit = limit
        self.window = 60  # seconds

    async def dispatch(self, request: Request, call_next):
        # Skip rate-limiting for health/metrics
        if request.url.path in ("/health", "/metrics"):
            return await call_next(request)

        ip = (
            request.headers.get("x-forwarded-for", "").split(",")[0].strip()
            or (request.client.host if request.client else "unknown")
        )
        key = f"rate:{ip}"

        try:
            redis = _get_redis()
            current = await redis.incr(key)
            if current == 1:
                await redis.expire(key, self.window)

            if current > self.limit:
                logger.warning("rate_limit_exceeded", ip=ip)
                return JSONResponse(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    content={"detail": "Too many requests. Please slow down."},
                    headers={"Retry-After": str(self.window)},
                )
        except Exception as e:
            # If Redis is down, don't block the request — fail open
            logger.error("rate_limiter_error", error=str(e))

        return await call_next(request)

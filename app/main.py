"""
app/main.py – FastAPI application factory
"""

import sentry_sdk
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.core.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from app.core.exceptions import register_exception_handlers
from app.api.v1.router import main_router
from app.utils.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = get_logger(None).bind(stage="startup")


def create_app() -> FastAPI:
    # ── Sentry (prod / staging only) ──────────
    if settings.SENTRY_DSN:
        sentry_sdk.init(
            dsn=settings.SENTRY_DSN,
            environment=settings.APP_ENV,
            traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        )

    app = FastAPI(
        title="Vision AI API",
        version="1.0.0",
        docs_url="/docs" if not settings.is_prod else None,   # hide Swagger in prod
        redoc_url="/redoc" if not settings.is_prod else None,
        openapi_url="/openapi.json" if not settings.is_prod else None,
        debug=settings.DEBUG,
    )

    # ── Middleware (order matters) ─────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(RateLimitMiddleware, limit=settings.RATE_LIMIT_PER_MINUTE)
    app.add_middleware(RequestLoggingMiddleware)

    # ── Exception handlers ────────────────────
    register_exception_handlers(app)

    # ── Routers ───────────────────────────────
    app.include_router(main_router)

    # ── Prometheus metrics ────────────────────
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")

    # ── Health check ──────────────────────────
    @app.get("/health", tags=["health"])
    async def health():
        return {"status": "ok", "env": settings.APP_ENV}

    logger.info("app_started", env=settings.APP_ENV, debug=settings.DEBUG)
    return app


app = create_app()

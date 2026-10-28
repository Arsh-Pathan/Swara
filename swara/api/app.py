"""FastAPI application factory for Swara."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from swara import __version__
from swara.api.routes import router
from swara.core.config import settings
from swara.engine import Swara
from swara.utils.logger import get_logger

logger = get_logger("swara.api")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application lifecycle: preloads models on startup and cleans up on shutdown."""
    if not hasattr(app.state, "engine") or app.state.engine is None:
        logger.info("Initializing Swara engine on API server startup...")
        engine = Swara(config=settings)
        app.state.engine = engine
        logger.info("Swara speech engine successfully loaded and ready for traffic.")
    yield
    logger.info("Shutting down Swara server...")


def create_app() -> FastAPI:
    """Builds and configures the FastAPI application instance."""
    app = FastAPI(
        title="Swara: Speaker Intelligence AI",
        description="Production-ready, ultra-low-latency, lightweight local speaker intelligence engine. CSKY - 'Every Voice Has an Identity.'",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(router)

    # Global Exception Handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled error processing {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": "InternalServerError", "detail": str(exc)},
        )

    return app


app = create_app()

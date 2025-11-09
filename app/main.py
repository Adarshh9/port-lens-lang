"""
Main FastAPI application entry point.
Initializes the RAG + LLM system.
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.logger import logger
from app.router import AppRouter
from app.api.routes import router as api_router, init_routes

# Global router
_app_router: AppRouter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _app_router

    # Startup
    logger.info("RAG + LLM System starting up")
    _app_router = AppRouter()
    init_routes(_app_router.graph_builder, _app_router.indexer)
    logger.info("System ready for requests")

    yield

    # Shutdown
    logger.info("RAG + LLM System shutting down")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Returns:
        Configured FastAPI instance
    """
    app = FastAPI(
        title="RAG + LLM System",
        description="Production-ready Retrieval Augmented Generation with LLM",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routes
    app.include_router(api_router)

    # Health check
    @app.get("/")
    async def root():
        return {
            "message": "RAG + LLM System API",
            "version": "1.0.0",
            "status": "running",
        }

    logger.info("FastAPI application created successfully")
    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_level=settings.log_level.lower(),
    )

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.logger import logger
from app.router import AppRouter
from app.api.routes import router as api_router, init_routes

# NEW: Import routing router and init function
from app.api.routes_with_routing import router as routing_router, init_routing

# Global router
_app_router: AppRouter = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global _app_router

    # Startup
    logger.info("RAG + LLM System starting up")
    _app_router = AppRouter()
    
    # Initialize legacy routes
    init_routes(_app_router.graph_builder, _app_router.indexer)
    
    # NEW: Initialize routing endpoints using the AppRouter's components
    if _app_router.model_config and _app_router.retriever:
        init_routing(_app_router.model_config, _app_router.retriever)
    
    logger.info("System ready for requests")

    yield

    # Shutdown
    logger.info("RAG + LLM System shutting down")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.
    """
    app = FastAPI(
        title="RAG + LLM System",
        description="Production-ready Retrieval Augmented Generation with LLM and Multi-Model Routing",
        version="1.1.0",
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
    
    # NEW: Include the smart routing router
    app.include_router(routing_router)

    # Health check
    @app.get("/")
    async def root():
        return {
            "message": "RAG + LLM System API",
            "version": "1.1.0",
            "status": "running",
            "features": ["langgraph", "multi-model-routing"]
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
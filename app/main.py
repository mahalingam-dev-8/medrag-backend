from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chat, documents, health, search, sessions
from app.config import get_settings
from app.core.embeddings import get_embedding_service
from app.db.database import create_tables, dispose_engine
from app.utils.exceptions import MedRAGError
from app.utils.logger import configure_logging, get_logger

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    logger.info("startup", env=settings.app_env)
    await create_tables()
    get_embedding_service().model  # warm up model before first request
    yield
    await dispose_engine()
    logger.info("shutdown")


app = FastAPI(
    title="Med RAG Assistant",
    description="Medical document retrieval-augmented generation API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(MedRAGError)
async def med_rag_error_handler(request: Request, exc: MedRAGError) -> JSONResponse:
    logger.error("application_error", error=str(exc), path=request.url.path)
    return JSONResponse(status_code=400, content={"detail": str(exc)})


# Register routers
app.include_router(health.router)
app.include_router(documents.router, prefix="/api/v1")
app.include_router(search.router, prefix="/api/v1")
app.include_router(chat.router, prefix="/api/v1")
app.include_router(sessions.router, prefix="/api/v1")

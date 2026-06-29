from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_llm_settings
from app.models import RuntimeConfigResponse
from app.routers.reviews import router as reviews_router
from app.routers.products import router as products_router

app = FastAPI(title="Automated PRD Review Framework", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(reviews_router)
app.include_router(products_router)


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config/runtime", response_model=RuntimeConfigResponse)
async def runtime_config() -> RuntimeConfigResponse:
    settings = get_llm_settings()
    return RuntimeConfigResponse(
        llm_enabled=settings.enabled,
        provider_name=settings.provider_name,
        base_url_configured=bool(settings.base_url),
        model=settings.model,
    )

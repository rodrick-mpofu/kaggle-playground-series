from fastapi import APIRouter, Depends
from api.schemas.irrigation import HealthResponse
from api.dependencies import ModelManager, get_model_manager

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(manager: ModelManager = Depends(get_model_manager)):
    return HealthResponse(
        status="ok",
        model_loaded=manager.is_loaded(),
        model_version=manager.model_version or "not loaded"
    )
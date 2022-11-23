from fastapi import APIRouter
from api.endpoints import predict, model

router = APIRouter()
router.include_router(model.router)
router.include_router(model.router)

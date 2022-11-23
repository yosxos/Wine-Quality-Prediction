from fastapi import APIRouter,Body
from fastapi import Query
from api.model.wine_model import WineModel
from typing import Optional

router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
)

@router.post("/")
async def predict(item_id:int, item:WineModel=Body(...,embed=True)):
    """Predict the score of the wine 
    Args:
        item_id (int): _description_
        item (WineModel, optional): _description_. Defaults to Body(...,embed=True).

    Returns:
        _type_: float 
    """
    return {}
@router.get("/")
async def perfect_wine():
    
    return 'wine'


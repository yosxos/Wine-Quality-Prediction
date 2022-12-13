from fastapi import APIRouter,Body
from fastapi import Query
from api.model.wine_model import WineModel
from typing import Optional
import pickle 


router = APIRouter(
    prefix="/api/predict",
    tags=["predict"],
)

@router.post("/")
async def predict(item:WineModel):
    """Predict the score of the wine given
    Args:
        item (WineModel): The wine the user wants to predict.

    Returns:
        _type_: float 
    """
    load_model=pickle.load(open('domaine/finalized_model.pkl','rb'))
    prediction=load_model.predict(item)
    return prediction
@router.get("/")
async def perfect_wine():
    """TO DO : call a methode to get the perfect wine

    Returns:
        _type_: Winemodel
    """
    
    return 'wine'


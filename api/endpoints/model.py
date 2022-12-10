from fastapi import APIRouter
from typing import Dict
from api.model.wine_model import WineModel
from domaine.ml_model import add_data , retrain_model
from fastapi.responses import FileResponse
router = APIRouter(
    prefix="/api/model",
    tags=["model"],
)

@router.get("/")
async def serialisation():
    return FileResponse("domaine/finalized_model.pkl")

@router.get("/description")
async def description()->Dict[str,str]:
    """Give info about the model

    Returns:
        Dict[str,str]: With shape;
    `
    {"Paramètres du modele":,"Performance":,"ect":,"":,}
    `
    """
    return "description"

@router.put("/")
async def add_wine(wine:WineModel):
    """Add a new wine to the data source

    Args:
        wine (WineModel): Needed wine data based on our module:
    fixed_acidity: float
    volatile_acidity:float
    citric_acid:float
    residual_sugar:float
    chlorides:float
    free_sulfur_dioxide:float
    total_sulfur_dioxide:float
    density:float
    pH:float=Field(...,gt=0,lt=14,description="Ph compris entre 0 et 14")
    sulphates:float
    alcohol:float
    quality:float
    """
    add_data(wine)  
    return wine

@router.post("/retrain")
async def retrain():
    """Retrain the model 
    """
    retrain_model()
    return "wine Added to the dataBase"

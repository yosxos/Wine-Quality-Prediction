from fastapi import APIRouter,Body
from fastapi import Query
from fastapi.encoders import jsonable_encoder
from api.model.wine_model import WineModel
import pandas as pd
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
    input_df = pd.DataFrame([item.dict()])
    input=input_df.drop(columns=['quality'],axis=1)
    #reshape data frame to fit our model
    input.columns = range(input.shape[1])
    prediction=load_model.predict(input)
    return prediction[0].item()
@router.get("/")
async def perfect_wine():
    """ Return the characteristics of a `perfect` wine 
    
    Returns :
        _type_ : WineModel
    """
    winesDf = pd.read_csv('datasource/Wines.csv')
    winesDf = winesDf.drop('Id',axis=1)
    bestWinesQuality = winesDf['quality'].max()
    winesDf = winesDf[winesDf['quality']==bestWinesQuality]
    winesDf = winesDf.drop('quality',axis=1)
    return(winesDf.mean())


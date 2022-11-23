from pydantic import BaseModel, Field
from typing import Optional

class WineModel(BaseModel):
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
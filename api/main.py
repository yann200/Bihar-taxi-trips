from fastapi import FastAPI, HTTPException
import uvicorn
import pickle
from pydantic import BaseModel, PrivateAttr, Field, PositiveFloat, computed_field
import  pandas as pd

import sys, os

parent_directory = os.path.abspath(os.path.join(os.getcwd(), '..'))

sys.path.append(parent_directory)

import common

app = FastAPI()


# project root
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_full_path(rel_path):
    return os.path.normpath(os.path.join(ROOT_DIR, rel_path))





# Charger le modèle sauvegardé
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Charger le modèle au démarrage de l'application
try:

    path = get_full_path("../models/taxi.model")

    model = load_model(path)

except Exception as e:
    model = None
    print(f"Error loading model: {e}")

# Définir un schéma pour les données entrantes
class PredictionRequest(BaseModel):
    hour: float
    weekday: float
    month: float



# Route pour vérifier le statut de l'API
@app.get("/")
def read_root():
    return {"message": "API is up and running!"}





# Route pour faire des prédictions
@app.post("/predict")
def predict(data: PredictionRequest):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convertir les données entrantes en DataFrame Pandas
        input_data = pd.DataFrame([data.dict()])

        # Faire une prédiction
        prediction = model.predict(input_data)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error making prediction: {e}")





if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0",
                port=8000, reload=True) 

"""
FastAPI Backend for Titanic Survival Prediction
Serves predictions using a trained Logistic Regression model
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

from schemas import PassengerInput, PassengerBatchInput, PredictionResponse, BatchPredictionResponse, HealthResponse

# Initialize FastAPI app
app = FastAPI(
    title="Titanic Survival Prediction API",
    description="A REST API for predicting Titanic passenger survival using Logistic Regression",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS for frontend usage
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model at startup
MODEL_PATH = Path(__file__).parent.parent / "model" / "logistic_model.joblib"
model = None


def load_model():
    """Load the trained model from disk"""
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")


@app.on_event("startup")
async def startup_event():
    """Load model on application startup"""
    try:
        load_model()
    except FileNotFoundError as e:
        print(f"Warning: {e}")
        print("Model will need to be loaded before making predictions")


def categorize_age(age: float) -> str:
    """Categorize age into groups matching the training data"""
    if age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teenager'
    elif age <= 35:
        return 'YoungAdult'
    elif age <= 55:
        return 'MiddleAged'
    else:
        return 'Senior'


def prepare_features(passenger: PassengerInput) -> pd.DataFrame:
    """Prepare input features for prediction"""
    # Calculate derived features
    family_size = passenger.sibsp + passenger.parch + 1
    is_alone = 1 if family_size == 1 else 0
    fare_per_person = passenger.fare / family_size if family_size > 0 else passenger.fare
    age_group = categorize_age(passenger.age)
    
    # Create DataFrame with all required features
    features = pd.DataFrame({
        'Pclass': [passenger.pclass],
        'Sex': [passenger.sex],
        'Age': [passenger.age],
        'SibSp': [passenger.sibsp],
        'Parch': [passenger.parch],
        'Fare': [passenger.fare],
        'Embarked': [passenger.embarked],
        'FamilySize': [family_size],
        'IsAlone': [is_alone],
        'FarePerPerson': [fare_per_person],
        'AgeGroup': [age_group]
    })
    
    return features


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - API health check"""
    return HealthResponse(
        status="healthy",
        message="Titanic Survival Prediction API is running",
        model_loaded=model is not None
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        message="API is running" if model is not None else "Model not loaded",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(passenger: PassengerInput):
    """
    Predict survival for a single passenger
    
    - **pclass**: Passenger class (1, 2, or 3)
    - **sex**: Gender ('male' or 'female')
    - **age**: Age in years
    - **sibsp**: Number of siblings/spouses aboard
    - **parch**: Number of parents/children aboard
    - **fare**: Ticket fare
    - **embarked**: Port of embarkation ('C', 'Q', or 'S')
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        # Prepare features
        features = prepare_features(passenger)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        
        # Prepare response
        survived = bool(prediction == 1)
        survival_probability = float(probability[1])
        
        return PredictionResponse(
            survived=survived,
            survival_probability=round(survival_probability, 4),
            confidence=round(max(probability) * 100, 2),
            message="Survived" if survived else "Did Not Survive"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(passengers: PassengerBatchInput):
    """
    Predict survival for multiple passengers
    
    Accepts a list of passengers and returns predictions for all.
    """
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please ensure the model file exists."
        )
    
    try:
        predictions = []
        
        for passenger in passengers.passengers:
            features = prepare_features(passenger)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]
            
            survived = bool(prediction == 1)
            survival_probability = float(probability[1])
            
            predictions.append(PredictionResponse(
                survived=survived,
                survival_probability=round(survival_probability, 4),
                confidence=round(max(probability) * 100, 2),
                message="Survived" if survived else "Did Not Survive"
            ))
        
        return BatchPredictionResponse(
            predictions=predictions,
            count=len(predictions)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded"
        )
    
    return {
        "model_type": type(model).__name__,
        "pipeline_steps": [step[0] for step in model.steps],
        "model_path": str(MODEL_PATH),
        "features": {
            "numerical": ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize', 'IsAlone', 'FarePerPerson'],
            "categorical": ['Pclass', 'Sex', 'Embarked', 'AgeGroup']
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

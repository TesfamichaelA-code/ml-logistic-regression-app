"""
Pydantic schemas for request/response validation
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Literal


class PassengerInput(BaseModel):
    """Schema for passenger input data"""
    
    pclass: int = Field(
        ...,
        ge=1,
        le=3,
        description="Passenger class (1 = First, 2 = Second, 3 = Third)"
    )
    sex: Literal["male", "female"] = Field(
        ...,
        description="Gender of the passenger"
    )
    age: float = Field(
        ...,
        ge=0,
        le=120,
        description="Age in years"
    )
    sibsp: int = Field(
        ...,
        ge=0,
        description="Number of siblings/spouses aboard"
    )
    parch: int = Field(
        ...,
        ge=0,
        description="Number of parents/children aboard"
    )
    fare: float = Field(
        ...,
        ge=0,
        description="Ticket fare"
    )
    embarked: Literal["C", "Q", "S"] = Field(
        ...,
        description="Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton)"
    )
    
    @field_validator('sex')
    @classmethod
    def validate_sex(cls, v: str) -> str:
        return v.lower()
    
    @field_validator('embarked')
    @classmethod
    def validate_embarked(cls, v: str) -> str:
        return v.upper()
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pclass": 1,
                    "sex": "female",
                    "age": 25,
                    "sibsp": 1,
                    "parch": 0,
                    "fare": 100.0,
                    "embarked": "S"
                }
            ]
        }
    }


class PassengerBatchInput(BaseModel):
    """Schema for batch passenger input"""
    
    passengers: List[PassengerInput] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of passengers to predict"
    )


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    
    survived: bool = Field(
        ...,
        description="Whether the passenger is predicted to survive"
    )
    survival_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Probability of survival (0-1)"
    )
    confidence: float = Field(
        ...,
        ge=0,
        le=100,
        description="Confidence percentage of the prediction"
    )
    message: str = Field(
        ...,
        description="Human-readable prediction result"
    )
    
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "survived": True,
                    "survival_probability": 0.85,
                    "confidence": 85.0,
                    "message": "Survived"
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Schema for batch prediction response"""
    
    predictions: List[PredictionResponse] = Field(
        ...,
        description="List of predictions for each passenger"
    )
    count: int = Field(
        ...,
        description="Number of predictions made"
    )


class HealthResponse(BaseModel):
    """Schema for health check response"""
    
    status: str = Field(
        ...,
        description="Health status of the API"
    )
    message: str = Field(
        ...,
        description="Status message"
    )
    model_loaded: bool = Field(
        ...,
        description="Whether the model is loaded"
    )

from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from predict import predict_data, get_model_info
from data import get_feature_names, get_feature_descriptions

app = FastAPI(
    title="Diabetes Progression Predictor",
    description=("Predict diabetes disease using Ridge Regression."),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class DiabetesData(BaseModel):
    """
    Ten normalised baseline measurements for a single patient.
    All values are mean-centred and scaled to unit variance.
    """
    age: float = Field(..., example=0.038,  description="Age (normalised)")
    sex: float = Field(..., example=0.051,  description="Sex (normalised)")
    bmi: float = Field(..., example=0.062,  description="Body Mass Index (normalised)")
    bp:  float = Field(..., example=0.022,  description="Average Blood Pressure (normalised)")
    s1:  float = Field(..., example=-0.044, description="Total serum cholesterol (normalised)")
    s2:  float = Field(..., example=-0.034, description="LDL cholesterol (normalised)")
    s3:  float = Field(..., example=-0.043, description="HDL cholesterol (normalised)")
    s4:  float = Field(..., example=-0.002, description="Total cholesterol / HDL ratio (normalised)")
    s5:  float = Field(..., example=0.019,  description="Log of serum triglycerides (normalised)")
    s6:  float = Field(..., example=-0.018, description="Blood sugar level (normalised)")


class DiabetesResponse(BaseModel):
    """Predicted disease progression score plus a descriptive risk band."""
    predicted_score: float
    risk_level: str


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded regression model."""
    model_type: str
    alpha: float
    n_features: int
    intercept: float

# helper methods

def to_features(d: DiabetesData) -> list:
    """Convert a DiabetesData object into a 2-D feature list for the model."""
    return [[d.age, d.sex, d.bmi, d.bp,
             d.s1,  d.s2,  d.s3,  d.s4, d.s5, d.s6]]


def score_to_risk(score: float) -> str:
    """
    Map a predicted score to a risk band using target-distribution terciles.
        Low    : score < 132
        Medium : 132 <= score < 213
        High   : score >= 213
    """
    if score < 132:
        return "Low"
    elif score < 213:
        return "Medium"
    return "High"


@app.get("/", status_code=status.HTTP_200_OK, tags=["Health"])
async def health_ping():
    """Health check — returns 200 OK when the service is running."""
    return {"status": "healthy"}


@app.post("/predict", response_model=DiabetesResponse, tags=["Prediction"])
async def predict_diabetes(patient: DiabetesData):
    """
    Predict a patient's diabetes disease progression score.

    Returns the predicted continuous score and a Low / Medium / High risk band.

    Raises:
        HTTPException 500: If the prediction pipeline fails.
    """
    try:
        score = float(predict_data(to_features(patient))[0])
        return DiabetesResponse(
            predicted_score=round(score, 2),
            risk_level=score_to_risk(score),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features", tags=["Info"])
async def list_features():
    """Return all expected input features with descriptions."""
    desc = get_feature_descriptions()
    return {
        "features": [
            {"name": n, "description": desc[n]}
            for n in get_feature_names()
        ]
    }


@app.get("/risk-bands", tags=["Info"])
async def risk_bands():
    """Return the score thresholds used to assign risk level labels."""
    return {
        "risk_bands": {
            "Low":    {"min": 25,  "max": 131},
            "Medium": {"min": 132, "max": 212},
            "High":   {"min": 213, "max": 346},
        },
        "note": "Bands are terciles of the training target distribution.",
    }


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Info"])
async def model_info():
    """
    Return metadata about the loaded Ridge Regression model.

    Raises:
        HTTPException 500: If model metadata cannot be retrieved.
    """
    try:
        return ModelInfoResponse(**get_model_info())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
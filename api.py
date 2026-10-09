"""FastAPI Backend Server for AI Risk Manager.
Exposes REST endpoints for real-time fraud risk scoring, anomaly detection, and SHAP explainability.
Deployable on Render as a persistent Python Web Service.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import uvicorn
import os
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.risk_engine import HybridRiskEngine
from src.explainability import XAIExplainer
from src.config import MODELS_DIR

app = FastAPI(
    title="AI Risk Manager API",
    description="Defense-only fraud detection and explainable risk evaluation API for Razorpay Buildathon",
    version="1.0.0"
)

# Enable CORS for Vercel frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows requests from Vercel deployments
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model references
engine: Optional[HybridRiskEngine] = None
explainer: Optional[XAIExplainer] = None

@app.on_event("startup")
def load_models():
    global engine, explainer
    print("Loading Hybrid Risk Engine and XAI Explainer...")
    try:
        engine = HybridRiskEngine.load_default()
        explainer = XAIExplainer.load_default()
        print("Models successfully loaded and ready for inference!")
    except Exception as e:
        print(f"Error loading models: {e}")

class TransactionRequest(BaseModel):
    trans_num: Optional[str] = "TXN_LIVE"
    cc_num: Optional[str] = "4532000011112222"
    amt: float = Field(..., gt=0, description="Transaction amount in USD/INR")
    category: str = Field(default="grocery_pos", description="Merchant category")
    hour: int = Field(default=14, ge=0, le=23, description="Hour of the day (0-23)")
    day_of_week: int = Field(default=2, ge=0, le=6, description="Day of the week (0=Mon, 6=Sun)")
    customer_age: float = Field(default=35.0, ge=18, le=100)
    haversine_distance_km: float = Field(default=5.0, ge=0)
    amt_to_user_avg_ratio: float = Field(default=1.0, gt=0)
    amt_to_cat_median_ratio: float = Field(default=1.0, gt=0)
    trans_velocity_1h: int = Field(default=0, ge=0)
    trans_velocity_24h: int = Field(default=1, ge=0)
    is_night_transaction: Optional[int] = None

@app.get("/")
def health_check():
    return {
        "status": "online",
        "service": "AI Risk Manager API",
        "models_loaded": engine is not None and explainer is not None
    }

@app.get("/api/scenarios")
def get_preset_scenarios():
    """Returns curated demo scenarios for the frontend."""
    return [
        {
            "id": "high_risk",
            "name": "High-Risk Fraud Scenario",
            "description": "Late-night large purchase, 7.8x spending surge, 1450 km terminal distance",
            "transaction": {
                "trans_num": "TXN_FRAUD_7849",
                "amt": 865.20,
                "category": "shopping_net",
                "hour": 3,
                "day_of_week": 5,
                "customer_age": 42.0,
                "haversine_distance_km": 1450.0,
                "amt_to_user_avg_ratio": 7.8,
                "amt_to_cat_median_ratio": 4.5,
                "trans_velocity_1h": 3,
                "trans_velocity_24h": 5
            }
        },
        {
            "id": "low_risk",
            "name": "Low-Risk Genuine Scenario",
            "description": "Routine daytime grocery checkout near billing residence",
            "transaction": {
                "trans_num": "TXN_GENUINE_1024",
                "amt": 38.50,
                "category": "grocery_pos",
                "hour": 12,
                "day_of_week": 1,
                "customer_age": 35.0,
                "haversine_distance_km": 3.2,
                "amt_to_user_avg_ratio": 0.82,
                "amt_to_cat_median_ratio": 0.91,
                "trans_velocity_1h": 0,
                "trans_velocity_24h": 1
            }
        },
        {
            "id": "moderate_risk",
            "name": "Moderate Anomaly Scenario",
            "description": "Late-night travel authorization with moderate geographical deviation",
            "transaction": {
                "trans_num": "TXN_MODERATE_3041",
                "amt": 220.0,
                "category": "travel",
                "hour": 1,
                "day_of_week": 6,
                "customer_age": 29.0,
                "haversine_distance_km": 450.0,
                "amt_to_user_avg_ratio": 2.8,
                "amt_to_cat_median_ratio": 1.8,
                "trans_velocity_1h": 1,
                "trans_velocity_24h": 2
            }
        }
    ]

@app.post("/api/predict")
def predict_transaction_risk(txn: TransactionRequest):
    """Calculates multi-signal hybrid risk score and SHAP attributions."""
    if not explainer or not engine:
        raise HTTPException(status_code=503, detail="Risk engines are still loading. Please retry.")

    data = txn.dict()
    # Auto-infer night flag if not provided
    if data.get("is_night_transaction") is None:
        data["is_night_transaction"] = 1 if (0 <= data["hour"] <= 5) else 0

    # Ensure required spatial and temporal keys are populated
    dist = data.get("haversine_distance_km", 5.0)
    hour = data.get("hour", 14)
    data["lat"] = data.get("lat", 32.7767)
    data["long"] = data.get("long", -96.7970)
    data["merch_lat"] = data.get("merch_lat", 32.7767 + (dist / 111.0))
    data["merch_long"] = data.get("merch_long", -96.7970)
    data["trans_date_trans_time"] = data.get("trans_date_trans_time", f"2026-08-27 {hour:02d}:30:00")

    try:
        explanation = explainer.explain_transaction(data)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=False)

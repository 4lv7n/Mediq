import os
from datetime import datetime
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mediq_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
KB_PATH = os.path.join(BASE_DIR, "MEDIQ_Disease_Knowledge_Base.xlsx")

FEATURES = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP", "HLA-B27",
    "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm", "C3", "C4"
]

KB_MAPPING = {
    "Rheumatoid Arthritis": "Rheumatoid Arthritis (RA)",
    "Ankylosing Spondylitis": "Ankylosing Spondylitis (AS)",
    "Sjogren Syndrome": "Sjogren's Syndrome (SS)",
    "Psoriatic Arthritis": "Psoriatic Arthritis (PsA)",
    "Reactive Arthritis": "Reactive Arthritis (ReA)",
    "Systemic Lupus Erythematosus": "Systemic Lupus Erythematosus (SLE)",
}

app = FastAPI(
    title="MEDIQ Prediction API",
    description="Educational/research API for the MEDIQ XGBoost model. Not for clinical diagnosis.",
    version="1.0.0",
)

# For development only. Restrict this list to your deployed frontend origin in production.
allowed_origins = os.getenv("MEDIQ_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == ["*"] else allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    knowledge_base = pd.read_excel(KB_PATH, sheet_name="Disease Knowledge Base")
    specialist_map = pd.read_excel(KB_PATH, sheet_name="Specialist Recommendation Map")
    explanation_templates = pd.read_excel(KB_PATH, sheet_name="Disease Explanation Templates")
except Exception as exc:
    raise RuntimeError(
        "Could not load model/encoder/knowledge base. Place mediq_model.pkl, "
        "label_encoder.pkl and MEDIQ_Disease_Knowledge_Base.xlsx beside main.py."
    ) from exc

if getattr(model, "n_features_in_", None) != len(FEATURES):
    raise RuntimeError(f"Expected model to use {len(FEATURES)} features; found {getattr(model, 'n_features_in_', None)}.")

def clean_value(value: Any):
    if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none"):
        return None
    return str(value)

def find_kb_row(disease: str):
    if "Disease" not in knowledge_base.columns:
        return None
    target = KB_MAPPING.get(disease, disease)
    series = knowledge_base["Disease"].astype(str).str.strip().str.casefold()
    rows = knowledge_base[series == target.strip().casefold()]
    if rows.empty:
        rows = knowledge_base[series == disease.strip().casefold()]
    return None if rows.empty else rows.iloc[0]

def disease_info(disease: str):
    if disease.strip().casefold() == "normal":
        return {
            "Status": "No disease class identified by the model",
            "Interpretation": (
                "The model classified the supplied input as Normal. "
                "This does not rule out disease or replace clinical assessment."
            )
        }

    row = find_kb_row(disease)
    info = {}
    row = find_kb_row(disease)
    info = {}
    if row is not None:
        for col in knowledge_base.columns:
            if col != "Disease":
                val = clean_value(row[col])
                if val:
                    info[col] = val
    for df, label in [(specialist_map, "Specialist recommendations"),
                      (explanation_templates, "Explanation templates")]:
        if "Disease" in df.columns:
            match = df[df["Disease"].astype(str).str.strip().str.casefold() == disease.strip().casefold()]
            if not match.empty:
                extra = {}
                for col in df.columns:
                    if col != "Disease":
                        val = clean_value(match.iloc[0][col])
                        if val:
                            extra[col] = val
                if extra:
                    info[label] = extra
    return info

class PatientInput(BaseModel):
    Age: float = Field(ge=1, le=120)
    Gender: int = Field(ge=0, le=1, description="Must use the model's original 0/1 encoding.")
    ESR: float = Field(ge=0)
    CRP: float = Field(ge=0)
    RF: float = Field(ge=0)
    anti_ccp: float = Field(ge=0, alias="Anti-CCP")
    hla_b27: int = Field(ge=0, le=1, alias="HLA-B27")
    ANA: int = Field(ge=0, le=1)
    anti_ro: int = Field(ge=0, le=1, alias="Anti-Ro")
    anti_la: int = Field(ge=0, le=1, alias="Anti-La")
    anti_dsdna: int = Field(ge=0, le=1, alias="Anti-dsDNA")
    anti_sm: int = Field(ge=0, le=1, alias="Anti-Sm")
    C3: float = Field(ge=0)
    C4: float = Field(ge=0)

    model_config = {"populate_by_name": True}

    def ordered_values(self):
        return [
            self.Age, self.Gender, self.ESR, self.CRP, self.RF,
            self.anti_ccp, self.hla_b27, self.ANA, self.anti_ro,
            self.anti_la, self.anti_dsdna, self.anti_sm, self.C3, self.C4
        ]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(FEATURES),
        "class_count": len(label_encoder.classes_),
    }

@app.get("/diseases")
def get_diseases():
    return {
        "diseases": [
            {"name": str(name), "information": disease_info(str(name))}
            for name in label_encoder.classes_
        ]
    }

@app.post("/predict")
def predict(payload: PatientInput):
    try:
        values = payload.ordered_values()
        patient = pd.DataFrame([values], columns=FEATURES)
        pred_encoded = int(model.predict(patient)[0])
        predicted_disease = str(label_encoder.inverse_transform([pred_encoded])[0])
        probabilities = model.predict_proba(patient)[0]
        model_classes = list(model.classes_)
        class_index = model_classes.index(pred_encoded)

        distribution = []
        for i, encoded_class in enumerate(model_classes):
            name = str(label_encoder.inverse_transform([int(encoded_class)])[0])
            distribution.append({
                "disease": name,
                "score_percent": round(float(probabilities[i]) * 100, 4)
            })
        distribution.sort(key=lambda x: x["score_percent"], reverse=True)

        shap_rows = []
        try:
            values_shap = np.asarray(shap.TreeExplainer(model).shap_values(patient))
            if values_shap.ndim == 3:
                contributions = values_shap[0, :, class_index]
            elif values_shap.ndim == 2:
                contributions = values_shap[0, :]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {values_shap.shape}")
            for feature, input_value, contribution in zip(FEATURES, values, contributions):
                shap_rows.append({
                    "feature": feature,
                    "input_value": input_value,
                    "contribution": float(contribution),
                    "absolute_contribution": abs(float(contribution)),
                    "direction": "positive" if float(contribution) > 0 else (
                        "negative" if float(contribution) < 0 else "neutral"
                    )
                })
            shap_rows.sort(key=lambda x: x["absolute_contribution"], reverse=True)
        except Exception as shap_exc:
            # Prediction remains available if the installed SHAP version is incompatible.
            shap_rows = []
            shap_warning = str(shap_exc)
        else:
            shap_warning = None

        result = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "predicted_disease": predicted_disease,
            "top_class_score_percent": round(float(np.max(probabilities)) * 100, 4),
            "probability_distribution": distribution,
            "input_features": dict(zip(FEATURES, values)),
            "shap_contributions": shap_rows,
            "disease_information": disease_info(predicted_disease),
            "shap_warning": shap_warning,
            "disclaimer": (
                "Educational research prototype only. This output is not a diagnosis. "
                "The top-class score is not necessarily a calibrated clinical probability. "
                "SHAP describes model behaviour, not medical causation."
            ),
        }
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")

@app.post("/report")
def report(payload: PatientInput):
    # Uses the same live prediction endpoint logic, returning a readable plain-text report.
    result = predict(payload)
    lines = [
        "MEDIQ — CLINICAL INTELLIGENCE ANALYSIS REPORT",
        "=" * 52,
        f"Generated: {result['generated_at']}",
        "",
        "IMPORTANT NOTICE",
        result["disclaimer"],
        "",
        "1. PREDICTION SUMMARY",
        f"Predicted class: {result['predicted_disease']}",
        f"Model-reported top-class score: {result['top_class_score_percent']:.2f}%",
        "",
        "2. PATIENT INPUT VALUES",
    ]
    lines += [f"- {k}: {v}" for k, v in result["input_features"].items()]
    lines += ["", "3. MODEL OUTPUT DISTRIBUTION (ALL CLASSES)"]
    lines += [f"- {r['disease']}: {r['score_percent']:.2f}%" for r in result["probability_distribution"]]
    lines += ["", "4. EXPLAINABILITY — SHAP CONTRIBUTIONS"]
    if result["shap_contributions"]:
        lines += [
            f"- {r['feature']}: contribution {r['contribution']:+.6f}; "
            f"absolute contribution {r['absolute_contribution']:.6f}; direction {r['direction']}"
            for r in result["shap_contributions"]
        ]
    else:
        lines.append("SHAP values are unavailable for this request.")
    lines += ["", "5. DISEASE KNOWLEDGE-BASE INFORMATION"]
    info = result["disease_information"]
    if info:
        for key, value in info.items():
            lines += ["", f"{key}:"]
            if isinstance(value, dict):
                lines += [f"- {subkey}: {subvalue}" for subkey, subvalue in value.items()]
            else:
                lines.append(str(value))
    else:
        lines.append("No matching disease information was found in the knowledge base.")
    lines += [
        "", "6. INTERPRETATION AND LIMITATIONS",
        "- The top-class score is not necessarily a calibrated clinical probability.",
        "- Model predictions may be wrong and should not be used alone for medical decisions.",
        "- SHAP contributions explain model behaviour, not medical causation.",
        "- Seek assessment from a qualified healthcare professional."
    ]
    return {"filename": "MEDIQ_Detailed_Analysis_Report.txt", "content": "\n".join(lines)}

@app.post("/ai-explanation")
def ai_explanation(payload: PatientInput):
    """Return the normal MEDIQ prediction plus an optional Gemini explanation.

    Gemini is only used when this endpoint is explicitly called. The core /predict
    endpoint remains independent of Gemini quotas and outages.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="AI explanation is not configured. Set GEMINI_API_KEY on the backend.",
        )

    # Reuse the actual model output; never ask Gemini to make or change the prediction.
    prediction = predict(payload)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        prompt_data = {
            "predicted_disease": prediction["predicted_disease"],
            "top_class_score_percent": prediction["top_class_score_percent"],
            "probability_distribution": prediction["probability_distribution"],
            "shap_contributions": prediction["shap_contributions"][:8],
            "disease_information": prediction["disease_information"],
        }
        prompt = (
            "You are the explanation component of MEDIQ, an educational research prototype, "
            "not a clinician. Explain the supplied machine-learning output in clear, cautious "
            "language for a student demonstration. The XGBoost model already made the prediction; "
            "do not change it, provide a new diagnosis, infer missing clinical facts, or claim "
            "that SHAP contributions are medical causation. Explain that positive/negative SHAP "
            "directions refer to model output only and that the score is not necessarily a calibrated "
            "clinical probability. Use only the supplied data and knowledge-base information. "
            "Do not repeat raw patient identifiers. Keep the response under 250 words and include "
            "a short limitations note. If information is missing, say so.\n\n"
            f"MEDIQ output (JSON): {__import__('json').dumps(prompt_data, ensure_ascii=False)}"
        )
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=500,
            ),
        )
        explanation = (response.text or "").strip()
        if not explanation:
            raise ValueError("Gemini returned an empty explanation.")

        return {
            "prediction": prediction,
            "ai_explanation": explanation,
            "ai_model": os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
            "disclaimer": prediction["disclaimer"],
        }
    except Exception as exc:
        # Keep the existing /predict endpoint available; return a useful error for this route.
        message = str(exc)
        lowered = message.lower()
        if "429" in message or "quota" in lowered or "resource_exhausted" in lowered:
            raise HTTPException(
                status_code=429,
                detail="Gemini quota/rate limit reached. The normal /predict endpoint still works; try the AI explanation later.",
            ) from exc
        if "404" in message or "not found" in lowered or "model" in lowered and "unsupported" in lowered:
            raise HTTPException(
                status_code=502,
                detail="Gemini model is unavailable for this API key/project. Check GEMINI_MODEL and model access.",
            ) from exc
        raise HTTPException(
            status_code=502,
            detail=f"Gemini explanation failed ({type(exc).__name__}). The normal /predict endpoint is unaffected.",
        ) from exc

"""import os
from datetime import datetime
from io import BytesIO
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "mediq_model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "label_encoder.pkl")
KB_PATH = os.path.join(BASE_DIR, "MEDIQ_Disease_Knowledge_Base.xlsx")

FEATURES = [
    "Age", "Gender", "ESR", "CRP", "RF", "Anti-CCP", "HLA-B27",
    "ANA", "Anti-Ro", "Anti-La", "Anti-dsDNA", "Anti-Sm", "C3", "C4"
]

KB_MAPPING = {
    "Rheumatoid Arthritis": "Rheumatoid Arthritis (RA)",
    "Ankylosing Spondylitis": "Ankylosing Spondylitis (AS)",
    "Sjogren Syndrome": "Sjogren's Syndrome (SS)",
    "Psoriatic Arthritis": "Psoriatic Arthritis (PsA)",
    "Reactive Arthritis": "Reactive Arthritis (ReA)",
    "Systemic Lupus Erythematosus": "Systemic Lupus Erythematosus (SLE)",
}

app = FastAPI(
    title="MEDIQ Prediction API",
    description="Educational/research API for the MEDIQ XGBoost model. Not for clinical diagnosis.",
    version="1.0.0",
)

# For development only. Restrict this list to your deployed frontend origin in production.
allowed_origins = os.getenv("MEDIQ_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if allowed_origins == ["*"] else allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    knowledge_base = pd.read_excel(KB_PATH, sheet_name="Disease Knowledge Base")
    specialist_map = pd.read_excel(KB_PATH, sheet_name="Specialist Recommendation Map")
    explanation_templates = pd.read_excel(KB_PATH, sheet_name="Disease Explanation Templates")
except Exception as exc:
    raise RuntimeError(
        "Could not load model/encoder/knowledge base. Place mediq_model.pkl, "
        "label_encoder.pkl and MEDIQ_Disease_Knowledge_Base.xlsx beside main.py."
    ) from exc

if getattr(model, "n_features_in_", None) != len(FEATURES):
    raise RuntimeError(f"Expected model to use {len(FEATURES)} features; found {getattr(model, 'n_features_in_', None)}.")

def clean_value(value: Any):
    if pd.isna(value) or str(value).strip().lower() in ("", "nan", "none"):
        return None
    return str(value)

def find_kb_row(disease: str):
    if "Disease" not in knowledge_base.columns:
        return None
    target = KB_MAPPING.get(disease, disease)
    series = knowledge_base["Disease"].astype(str).str.strip().str.casefold()
    rows = knowledge_base[series == target.strip().casefold()]
    if rows.empty:
        rows = knowledge_base[series == disease.strip().casefold()]
    return None if rows.empty else rows.iloc[0]

def disease_info(disease: str):
    if disease.strip().casefold() == "normal":
        return {
            "Status": "No disease class identified by the model",
            "Interpretation": (
                "The model classified the supplied input as Normal. "
                "This does not rule out disease or replace clinical assessment."
            )
        }

    row = find_kb_row(disease)
    info = {}
    row = find_kb_row(disease)
    info = {}
    if row is not None:
        for col in knowledge_base.columns:
            if col != "Disease":
                val = clean_value(row[col])
                if val:
                    info[col] = val
    for df, label in [(specialist_map, "Specialist recommendations"),
                      (explanation_templates, "Explanation templates")]:
        if "Disease" in df.columns:
            match = df[df["Disease"].astype(str).str.strip().str.casefold() == disease.strip().casefold()]
            if not match.empty:
                extra = {}
                for col in df.columns:
                    if col != "Disease":
                        val = clean_value(match.iloc[0][col])
                        if val:
                            extra[col] = val
                if extra:
                    info[label] = extra
    return info

class PatientInput(BaseModel):
    Age: float = Field(ge=1, le=120)
    Gender: int = Field(ge=0, le=1, description="Must use the model's original 0/1 encoding.")
    ESR: float = Field(ge=0)
    CRP: float = Field(ge=0)
    RF: float = Field(ge=0)
    anti_ccp: int = Field(ge=0, le=1, alias="Anti-CCP")
    hla_b27: int = Field(ge=0, le=1, alias="HLA-B27")
    ANA: int = Field(ge=0, le=1)
    anti_ro: int = Field(ge=0, le=1, alias="Anti-Ro")
    anti_la: int = Field(ge=0, le=1, alias="Anti-La")
    anti_dsdna: int = Field(ge=0, le=1, alias="Anti-dsDNA")
    anti_sm: int = Field(ge=0, le=1, alias="Anti-Sm")
    C3: float = Field(ge=0)
    C4: float = Field(ge=0)

    model_config = {"populate_by_name": True}

    def ordered_values(self):
        return [
            self.Age, self.Gender, self.ESR, self.CRP, self.RF,
            self.anti_ccp, self.hla_b27, self.ANA, self.anti_ro,
            self.anti_la, self.anti_dsdna, self.anti_sm, self.C3, self.C4
        ]

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "feature_count": len(FEATURES),
        "class_count": len(label_encoder.classes_),
    }

@app.get("/diseases")
def get_diseases():
    return {
        "diseases": [
            {"name": str(name), "information": disease_info(str(name))}
            for name in label_encoder.classes_
        ]
    }

@app.post("/predict")
def predict(payload: PatientInput):
    try:
        values = payload.ordered_values()
        patient = pd.DataFrame([values], columns=FEATURES)
        pred_encoded = int(model.predict(patient)[0])
        predicted_disease = str(label_encoder.inverse_transform([pred_encoded])[0])
        probabilities = model.predict_proba(patient)[0]
        model_classes = list(model.classes_)
        class_index = model_classes.index(pred_encoded)

        distribution = []
        for i, encoded_class in enumerate(model_classes):
            name = str(label_encoder.inverse_transform([int(encoded_class)])[0])
            distribution.append({
                "disease": name,
                "score_percent": round(float(probabilities[i]) * 100, 4)
            })
        distribution.sort(key=lambda x: x["score_percent"], reverse=True)

        shap_rows = []
        try:
            values_shap = np.asarray(shap.TreeExplainer(model).shap_values(patient))
            if values_shap.ndim == 3:
                contributions = values_shap[0, :, class_index]
            elif values_shap.ndim == 2:
                contributions = values_shap[0, :]
            else:
                raise ValueError(f"Unexpected SHAP output shape: {values_shap.shape}")
            for feature, input_value, contribution in zip(FEATURES, values, contributions):
                shap_rows.append({
                    "feature": feature,
                    "input_value": input_value,
                    "contribution": float(contribution),
                    "absolute_contribution": abs(float(contribution)),
                    "direction": "positive" if float(contribution) > 0 else (
                        "negative" if float(contribution) < 0 else "neutral"
                    )
                })
            shap_rows.sort(key=lambda x: x["absolute_contribution"], reverse=True)
        except Exception as shap_exc:
            # Prediction remains available if the installed SHAP version is incompatible.
            shap_rows = []
            shap_warning = str(shap_exc)
        else:
            shap_warning = None

        result = {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "predicted_disease": predicted_disease,
            "top_class_score_percent": round(float(np.max(probabilities)) * 100, 4),
            "probability_distribution": distribution,
            "input_features": dict(zip(FEATURES, values)),
            "shap_contributions": shap_rows,
            "disease_information": disease_info(predicted_disease),
            "shap_warning": shap_warning,
            "disclaimer": (
                "Educational research prototype only. This output is not a diagnosis. "
                "The top-class score is not necessarily a calibrated clinical probability. "
                "SHAP describes model behaviour, not medical causation."
            ),
        }
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {type(exc).__name__}: {exc}")

@app.post("/report")
def report(payload: PatientInput):
    # Uses the same live prediction endpoint logic, returning a readable plain-text report.
    result = predict(payload)
    lines = [
        "MEDIQ — CLINICAL INTELLIGENCE ANALYSIS REPORT",
        "=" * 52,
        f"Generated: {result['generated_at']}",
        "",
        "IMPORTANT NOTICE",
        result["disclaimer"],
        "",
        "1. PREDICTION SUMMARY",
        f"Predicted class: {result['predicted_disease']}",
        f"Model-reported top-class score: {result['top_class_score_percent']:.2f}%",
        "",
        "2. PATIENT INPUT VALUES",
    ]
    lines += [f"- {k}: {v}" for k, v in result["input_features"].items()]
    lines += ["", "3. MODEL OUTPUT DISTRIBUTION (ALL CLASSES)"]
    lines += [f"- {r['disease']}: {r['score_percent']:.2f}%" for r in result["probability_distribution"]]
    lines += ["", "4. EXPLAINABILITY — SHAP CONTRIBUTIONS"]
    if result["shap_contributions"]:
        lines += [
            f"- {r['feature']}: contribution {r['contribution']:+.6f}; "
            f"absolute contribution {r['absolute_contribution']:.6f}; direction {r['direction']}"
            for r in result["shap_contributions"]
        ]
    else:
        lines.append("SHAP values are unavailable for this request.")
    lines += ["", "5. DISEASE KNOWLEDGE-BASE INFORMATION"]
    info = result["disease_information"]
    if info:
        for key, value in info.items():
            lines += ["", f"{key}:"]
            if isinstance(value, dict):
                lines += [f"- {subkey}: {subvalue}" for subkey, subvalue in value.items()]
            else:
                lines.append(str(value))
    else:
        lines.append("No matching disease information was found in the knowledge base.")
    lines += [
        "", "6. INTERPRETATION AND LIMITATIONS",
        "- The top-class score is not necessarily a calibrated clinical probability.",
        "- Model predictions may be wrong and should not be used alone for medical decisions.",
        "- SHAP contributions explain model behaviour, not medical causation.",
        "- Seek assessment from a qualified healthcare professional."
    ]
    return {"filename": "MEDIQ_Detailed_Analysis_Report.txt", "content": "\n".join(lines)}
"""

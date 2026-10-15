# MEDIQ Python API backend

This backend loads the existing MEDIQ XGBoost model, label encoder, and disease knowledge base.

## Files required beside `main.py`
- `mediq_model.pkl`
- `label_encoder.pkl`
- `MEDIQ_Disease_Knowledge_Base.xlsx`

## Run locally
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://127.0.0.1:8000/docs` to test the API.

## Endpoints
- `GET /health`
- `GET /diseases`
- `POST /predict`
- `POST /report`

## Example JSON body for `/predict` or `/report`
```json
{
  "Age": 30,
  "Gender": 0,
  "ESR": 20,
  "CRP": 5,
  "RF": 10,
  "Anti-CCP": 0,
  "HLA-B27": 0,
  "ANA": 0,
  "Anti-Ro": 0,
  "Anti-La": 0,
  "Anti-dsDNA": 0,
  "Anti-Sm": 0,
  "C3": 100,
  "C4": 20
}
```

**Important:** Gender and binary marker encoding must match the original model training data. Confirm the training mapping before interpreting outputs. This is an educational research prototype, not a diagnostic tool.

## Deployment
Deploy this folder to a Python-compatible host such as Render. Set `MEDIQ_ALLOWED_ORIGINS` to the deployed frontend origin before production use. Keep model/data files private to the backend. Do not commit API keys.

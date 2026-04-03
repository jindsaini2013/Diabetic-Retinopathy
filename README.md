# Diabetic Retinopathy Detection System

A full-stack AI-powered web application for automated Diabetic Retinopathy grading from retinal fundus photographs.

## Stack

- FastAPI backend at the repository root
- Vite + React frontend in `dr-detection-frontend`
- PyTorch model weights in `best_dr_fixed_model.pth`
- ONNX runtime model in `best_dr_fixed_model.onnx` for Vercel deployment

## Vercel Deployment

This repository is configured for a single Vercel project using two services:

- `web` serves the Vite frontend at `/`
- `api` serves the FastAPI backend at `/api`

The frontend already defaults to calling the backend through `/api`, so no Docker or Jenkins setup is needed.
To fit Vercel's function storage limits, the deployed API uses ONNX Runtime instead of PyTorch.

### Deploy steps

1. Push this repository to GitHub.
2. Import the repo into Vercel.
3. In Project Settings, set the Framework Preset to `Services`.
4. Deploy.

### Local development

Backend:

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Frontend:

```bash
cd dr-detection-frontend
npm install
npm run dev
```

If you run the frontend locally against the local backend, set `VITE_API_URL=http://localhost:8000`.

## Model

- Architecture: EfficientDRNet with CBAM attention
- Dataset: APTOS 2019 Blindness Detection
- Classes: `0` No DR, `1` Mild DR, `2` Moderate DR, `3` Severe DR
- Explainability: Grad-CAM remains a local-only workflow; the Vercel API serves predictions through ONNX Runtime

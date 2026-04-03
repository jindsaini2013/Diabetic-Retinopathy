# Diabetic Retinopathy Detection System

A full-stack AI-powered web application for automated Diabetic Retinopathy grading from retinal fundus photographs.

## Stack

- FastAPI backend at the repository root
- Vite + React frontend in `dr-detection-frontend`
- PyTorch model weights in `best_dr_fixed_model.pth`

## Deployment Plan

To keep the API fully working, including Grad-CAM, and avoid serverless cold-start and storage limits:

- Deploy the frontend to Vercel
- Deploy the FastAPI backend to Render as a web service

This avoids Docker and Jenkins while keeping the API continuously available as a normal Python service.

### Backend on Render

The repository includes [render.yaml](/Users/jindsaini/Desktop/Diabetic-Retinopathy/render.yaml) for the API service.

1. Import this repo into Render.
2. Create a `Web Service`.
3. Use the root directory of the repository.
4. Render will use:
   `buildCommand: pip install -r requirements.txt`
   `startCommand: uvicorn main:app --host 0.0.0.0 --port $PORT`
5. After deploy, note your backend URL, for example:
   `https://diabetic-retinopathy-api.onrender.com`

### Frontend on Vercel

The frontend includes [dr-detection-frontend/vercel.json](/Users/jindsaini/Desktop/Diabetic-Retinopathy/dr-detection-frontend/vercel.json) for SPA routing.

1. Import this same repo into Vercel.
2. Set the project `Root Directory` to `dr-detection-frontend`.
3. Add an environment variable:
   `VITE_API_URL=https://your-render-backend-url.onrender.com`
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

If you run the frontend locally against the local backend, it already defaults to `http://localhost:8000`.

## Model

- Architecture: EfficientDRNet with CBAM attention
- Dataset: APTOS 2019 Blindness Detection
- Classes: `0` No DR, `1` Mild DR, `2` Moderate DR, `3` Severe DR
- Explainability: Grad-CAM heatmap visualization

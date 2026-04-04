# Diabetic Retinopathy Detection System

A full-stack AI-powered web application for automated Diabetic Retinopathy grading from retinal fundus photographs.

## Stack

- FastAPI backend at the repository root
- Vite + React frontend in `dr-detection-frontend`
- PyTorch model weights in `best_dr_fixed_model.pth`

## Deployment Plan

To keep the API fully working, including Grad-CAM, and avoid serverless cold-start and storage limits:

- Deploy the frontend to Vercel
- Deploy the FastAPI backend to a Hugging Face Docker Space

This keeps the frontend and backend separate. On free Hugging Face hardware, the backend can still sleep when idle.

### Backend on Hugging Face Spaces

The repository includes:

- [deploy/hf-space/README.md](/Users/jindsaini/Desktop/Diabetic-Retinopathy/deploy/hf-space/README.md)
- [deploy/hf-space/Dockerfile](/Users/jindsaini/Desktop/Diabetic-Retinopathy/deploy/hf-space/Dockerfile)
- [.github/workflows/sync-hf-space.yml](/Users/jindsaini/Desktop/Diabetic-Retinopathy/.github/workflows/sync-hf-space.yml)

Setup:

1. Create a new Hugging Face Space with SDK `Docker`.
2. Add a GitHub secret named `HF_TOKEN`.
3. Add a GitHub repository variable named `HF_SPACE_ID`.
   Example: `your-hf-username/diabetic-retinopathy-api`
4. Every push to `main` will sync the backend files to the Space automatically.

After the Space builds, your backend URL will look like:

`https://your-hf-space-subdomain.hf.space`

### Frontend on Vercel

The frontend includes [dr-detection-frontend/vercel.json](/Users/jindsaini/Desktop/Diabetic-Retinopathy/dr-detection-frontend/vercel.json) for SPA routing.

1. Import this same repo into Vercel.
2. Set the project `Root Directory` to `dr-detection-frontend`.
3. Add an environment variable:
   `VITE_API_URL=https://your-hf-space-subdomain.hf.space`
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

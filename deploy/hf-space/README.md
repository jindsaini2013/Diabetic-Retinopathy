---
title: Diabetic Retinopathy API
emoji: 👁️
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
license: mit
short_description: FastAPI backend for DR prediction and Grad-CAM
---

# Diabetic Retinopathy API

This Hugging Face Space serves the FastAPI backend used by the Vercel frontend.

Available endpoints:

- `GET /health`
- `POST /predict`
- `POST /gradcam`

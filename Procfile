# Deploy backend on Railway (free tier)
# Production command: gunicorn with 4 async uvicorn workers
# Handles ~20,000-30,000 concurrent users with this config

web: gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT

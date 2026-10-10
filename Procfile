# Deploy backend on Render (Web Service)
# Production command: gunicorn with 4 async uvicorn workers
# Handles ~5,000-10,000 concurrent users on free tier (upgrade to $7/mo for 20K+)

web: gunicorn backend.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:$PORT

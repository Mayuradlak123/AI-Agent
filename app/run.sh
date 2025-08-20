#!/bin/bash

echo "🚀 Starting the FastAPI server..."
echo "🔍 Hosting on http://127.0.0.1:8000"
echo "📜 Swagger docs available at http://127.0.0.1:8000/docs"
echo "🛠️  Using: main:app with hot-reload enabled"

# Run the FastAPI app
python -m uvicorn main:app --host 127.0.0.1 --port 8080 --reload

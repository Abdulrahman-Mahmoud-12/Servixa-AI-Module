# Servixa AI

Servixa AI is a FastAPI-based microservice for AI-powered document and review processing. It currently provides:

- ID verification from Egyptian National ID card images
- Review analysis for authenticity and sentiment
- Health monitoring and structured API responses

## Features

- Upload and verify ID card images using computer vision and OCR
- Extract and validate key fields from ID documents
- Analyze customer reviews for sentiment and authenticity signals
- Expose a clean REST API for integration into a larger backend system

## Project Structure

- app/main.py - FastAPI application entry point
- app/api/ - API routers for each module
- app/services/ - business logic services
- app/modules/ - model and inference implementations
- app/schemas/ - request/response models
- tests/ - automated tests

## Requirements

Python 3.10+ is recommended.

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use .venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Configuration

Create a .env file in the project root with values such as:

```env
APP_NAME=Servixa AI
APP_VERSION=0.1.0
HOST=0.0.0.0
PORT=8000
DEBUG=false

GROQ_API_KEY=your_api_key_here
```

The application also uses default paths for storage and model files under the app/storage and app/modules directories.

## Running the Server

Start the API with:

```bash
python -m app.main
```

Or with Uvicorn directly:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

### Health Check

- GET /router/health

### ID Verification

- POST /router/id-verification/verify
  - Accepts an uploaded image file
  - Returns extracted and validated ID information

### Review Analysis

- POST /router/review-analysis/analyze
  - Accepts a review request payload
  - Returns structured analysis results

### Swagger Docs

- http://localhost:8000/router/docs
- http://localhost:8000/router/redoc

## Testing

Run tests with:

```bash
pytest
```

## Notes

This project is designed to be extended with additional modules such as pricing prediction and chatbot features in future iterations.

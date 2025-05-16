# Annotated Dataset Generator for Fine-tuning

A tool for generating annotated datasets for fine-tuning machine learning models.

## Prerequisites

- Redis
- Node.js and npm
- Python 3.x

## Installation

### 1. Install Redis

```bash
brew install redis
```

### 2. Install Node.js and npm

```bash
brew install node
```

### 3. Frontend Setup

```bash
cd frontend
npm install
```

### 4. Backend Setup

```bash
cd backend
python3 -m venv env
source env/bin/activate
pip3 install --upgrade --force-reinstall -r requirements.txt
```

### 5. Environment Setup

Create a `.env` file in the `backend` directory with the following variables:

```bash
# LLM API Credentials
LLM_API_KEY=your_api_key_here
LLM_API_BASE=your_api_base_url_here
LLM_MODEL=your_model_name_here

# VLM API Credentials
VLM_API_KEY=your_api_key_here
VLM_API_BASE=your_api_base_url_here
VLM_MODEL=your_model_name_here

# Environment variables for web search integration
UNSPLASH_ACCESS_KEY=your_unsplash_access_key_here
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_SEARCH_ENGINE_ID=your_google_search_engine_id_here

```

Replace the values with your actual credentials. Never commit the `.env` file to version control.

## Running the Application

1. Start Redis server:

```bash
redis-server
```

2. Start the frontend development server:

```bash
cd frontend
npm run dev
```

3. Start the backend server:

```bash
cd backend
source env/bin/activate
uvicorn main:app --reload
```

## Project Structure

- `frontend/` - React-based web interface
- `backend/` - FastAPI server and data processing logic

## Web Search Integration Setup

This application supports retrieving images from multiple sources:

- Bing (web scraping)
- Google [using Google Custom Search API] ( yetto be added)
- Unsplash (using Unsplash API)

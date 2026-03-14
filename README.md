# Domain-Specific Text Generator with Fine-Tuned Transformers

A beginner-friendly project for fine-tuning GPT-2 on domain-specific text and serving generation through a production-ready FastAPI backend plus a static web frontend.

## Overview

This repository shows a full practical workflow:
- Fine-tune `gpt2` on your own dataset (`data/domain_dataset.txt`)
- Run an API for inference (`/generate`)
- Use a simple responsive frontend (`app/static/*`)
- Deploy quickly on Render (`render.yaml`)

If `fine_tuned_model/` exists, the app loads it. If not, it gracefully falls back to base `gpt2` so the project still runs.

## Features

- FastAPI backend with clear API endpoints
- Static frontend served by the backend
- Health check endpoint for deployment monitoring
- Fallback model behavior (fine-tuned model -> base GPT-2)
- Training script using Hugging Face `Trainer`
- Render deployment configuration included

## Project Structure

```text
.
├── app/
│   ├── app.py                 # FastAPI backend + static file serving
│   └── static/
│       ├── index.html         # Frontend UI
│       ├── styles.css         # Frontend styling
│       └── script.js          # Frontend API calls
├── data/
│   └── domain_dataset.txt     # Domain training text (one example per line)
├── scripts/
│   └── fine_tune.py           # GPT-2 fine-tuning script
├── requirements.txt
├── render.yaml
└── README.md
```

## Local Setup

1. **Clone and enter the project**
   ```bash
   git clone <your-repo-url>
   cd Domain-Specific-Text-Generator-with-Fine-Tuned-Transformers
   ```

2. **Create and activate a virtual environment (recommended)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # .venv\Scripts\activate    # Windows PowerShell
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Training Instructions

Fine-tune GPT-2 on your dataset:

```bash
python scripts/fine_tune.py \
  --data_file data/domain_dataset.txt \
  --output_dir fine_tuned_model \
  --epochs 3 \
  --block_size 128
```

After training, the model and tokenizer are saved to `fine_tuned_model/`.

## Run Locally

Start the FastAPI app:

```bash
uvicorn app.app:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- Frontend: `http://localhost:8000/`
- API docs: `http://localhost:8000/docs`

## API Usage

### Health Check

```bash
curl http://localhost:8000/health
```

Example response:

```json
{
  "status": "ok",
  "model_source": "not_loaded",
  "fallback_model": "gpt2"
}
```

### Generate Text

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a concise technical note about transformer fine-tuning:",
    "max_length": 100
  }'
```

Example response:

```json
{
  "prompt": "Write a concise technical note about transformer fine-tuning:",
  "generated_text": "...",
  "model_source": "gpt2"
}
```

## Frontend Usage

1. Open `http://localhost:8000/`
2. Enter a prompt
3. Adjust max length
4. Click **Generate**
5. Read output in the generated text panel

## Render Deployment

This project includes `render.yaml` for easy deployment.

### Option A: Blueprint Deploy (recommended)

1. Push this repository to GitHub
2. In Render, choose **New +** -> **Blueprint**
3. Connect the repo and deploy

Render uses:
- `buildCommand`: `pip install -r requirements.txt`
- `startCommand`: `uvicorn app.app:app --host 0.0.0.0 --port $PORT`

### Option B: Manual Web Service

If you create the service manually in Render dashboard:
- **Environment**: Python
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.app:app --host 0.0.0.0 --port $PORT`
- **Python Version**: 3.10+

## Fallback Model Behavior

At startup the backend checks for `fine_tuned_model/`:
- Present -> loads your fine-tuned model
- Missing -> loads base `gpt2`

This ensures local development and deployment still work before training is complete.

## Future Improvements

- Add streaming token output for better UX
- Add evaluation scripts and benchmark reports
- Add request authentication/rate limiting for public API usage
- Add automated tests for API endpoints
- Add CI for linting and deployment checks

## Acknowledgements

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [PyTorch](https://pytorch.org/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Render](https://render.com/)

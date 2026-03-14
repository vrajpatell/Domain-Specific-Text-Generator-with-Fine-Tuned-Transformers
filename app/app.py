"""FastAPI app for domain-specific text generation."""

from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MODEL_DIR = Path("fine_tuned_model")
FALLBACK_MODEL = "gpt2"

# Lazy-loaded model state so the web server can start quickly on deployment platforms.
tokenizer: GPT2TokenizerFast | None = None
model: GPT2LMHeadModel | None = None
loaded_model_source: str | None = None
load_error: str | None = None
model_lock = Lock()
device = "cuda" if torch.cuda.is_available() else "cpu"


app = FastAPI(title="Domain-Specific Text Generator API")

# Allows easier local development and deployment behind proxies.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt text used for generation")
    max_length: int = Field(100, ge=20, le=512, description="Maximum output token length")


class GenerateResponse(BaseModel):
    prompt: str
    generated_text: str
    model_source: str


def _load_model_and_tokenizer() -> None:
    """Load the model once: fine-tuned model first, then base GPT-2 fallback."""
    global tokenizer, model, loaded_model_source, load_error

    if tokenizer is not None and model is not None:
        return

    with model_lock:
        if tokenizer is not None and model is not None:
            return

        model_source = str(MODEL_DIR) if MODEL_DIR.exists() else FALLBACK_MODEL
        try:
            local_tokenizer = GPT2TokenizerFast.from_pretrained(model_source)
            local_model = GPT2LMHeadModel.from_pretrained(model_source)

            # GPT-2 has no pad token by default.
            local_tokenizer.pad_token = local_tokenizer.eos_token
            local_model.config.pad_token_id = local_tokenizer.eos_token_id

            local_model.to(device)
            local_model.eval()

            tokenizer = local_tokenizer
            model = local_model
            loaded_model_source = model_source
            load_error = None
        except Exception as exc:  # Keep API alive and report cleanly.
            load_error = str(exc)
            tokenizer = None
            model = None
            loaded_model_source = None


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/index.html")
def index_alias() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "model_source": loaded_model_source or "not_loaded",
        "fallback_model": FALLBACK_MODEL,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    prompt = payload.prompt.strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty.")

    _load_model_and_tokenizer()
    if tokenizer is None or model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model is unavailable right now. Startup load failed: {load_error}",
        )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=payload.max_length,
            do_sample=True,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return GenerateResponse(
        prompt=prompt,
        generated_text=generated_text,
        model_source=loaded_model_source or FALLBACK_MODEL,
    )

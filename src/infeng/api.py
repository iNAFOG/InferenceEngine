import json
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from .engine import InferenceEngine
from .sampler import validate_sampling_params
from .schemas import BatchGenerateRequest, BatchGenerateResponse, GenerateRequest, GenerateResponse

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once when the server starts, not when this file is imported
    state["engine"] = InferenceEngine()
    yield
    state.clear()


app = FastAPI(title="InfEng", version="0.2.0", lifespan=lifespan)

# one model on one device: let one request use it at a time.
# fastapi runs these sync endpoints in a thread pool, so without the lock two
# requests would run forward passes on the same model at the same time.
engine_lock = threading.Lock()


def get_engine() -> InferenceEngine:
    return state["engine"]


@app.get("/health")
def health() -> dict:
    engine = get_engine()
    return {
        "status": "ok",
        "model": engine.config.model_name,
        "device": engine.device,
        "dtype": str(engine.dtype).replace("torch.", ""),
        "kv_cache": engine.config.use_kv_cache,
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        with engine_lock:
            result = get_engine().generate(req.prompt, req.to_sampling_params())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return GenerateResponse(**result)


@app.post("/generate/batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest) -> BatchGenerateResponse:
    try:
        with engine_lock:
            results = get_engine().generate_batch(req.prompts, req.to_sampling_params())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return BatchGenerateResponse(results=[GenerateResponse(**r) for r in results])


@app.post("/generate/stream")
def generate_stream(req: GenerateRequest):
    params = req.to_sampling_params()
    try:
        # validate before streaming starts, once the 200 header is sent we can't return a 400
        validate_sampling_params(params)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    def ndjson_lines():
        # StreamingResponse needs str/bytes, so every event is one json object per line
        with engine_lock:
            for event in get_engine().stream_generate(req.prompt, params):
                yield json.dumps(event) + "\n"

    return StreamingResponse(ndjson_lines(), media_type="application/x-ndjson")

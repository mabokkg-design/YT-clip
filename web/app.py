"""
YT AI Clip - Web app backend
Run with: uvicorn app:app --reload
"""

import os
import json
import httpx
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

REKA_BASE_URL = "https://vision-agent.api.reka.ai/v1/clips"

# Shared client — one connection pool for the lifetime of the process
_http: httpx.AsyncClient


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _http
    _http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=600.0))
    yield
    await _http.aclose()


app = FastAPI(title="YT AI Clip", lifespan=lifespan)


class GenerateRequest(BaseModel):
    video_url: str
    prompt: str = "Create an engaging video clip highlighting the best moments"
    template: str = "moments"
    num_generations: int = 3  # Reka max is 3
    min_duration: int = 0
    max_duration: int = 60
    subtitles: bool = True
    aspect_ratio: str = "9:16"
    resolution: int = 1080   # 240 | 360 | 480 | 720 | 1080
    layout: str = "ai"       # "ai" | "fit"


def get_api_key() -> str:
    key = os.environ.get("REKA_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="REKA_API_KEY environment variable is not set on the server.")
    return key


def reka_headers(api_key: str) -> dict:
    return {"X-Api-Key": api_key, "Content-Type": "application/json"}


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    api_key = get_api_key()

    payload = {
        "video_urls": [req.video_url],
        "prompt": req.prompt,
        "generation_config": {
            "template": req.template,
            "num_generations": min(req.num_generations, 3),
            "min_duration_seconds": req.min_duration,
            "max_duration_seconds": req.max_duration,
        },
        "rendering_config": {
            "subtitles": req.subtitles,
            "aspect_ratio": req.aspect_ratio,
            "resolution": req.resolution,
            "layout": req.layout,
        },
        "stream": True,
    }

    async def event_stream():
        last_data = None
        try:
            async with _http.stream("POST", REKA_BASE_URL, json=payload, headers=reka_headers(api_key)) as response:
                if response.status_code != 200:
                    body = await response.aread()
                    yield f"data: {json.dumps({'error': f'API {response.status_code}: {body.decode()}'})}\n\n"
                    return
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str != last_data:
                            yield f"data: {data_str}\n\n"
                            last_data = data_str
        except Exception as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/clips")
async def list_clips():
    api_key = get_api_key()
    response = await _http.get(REKA_BASE_URL, headers=reka_headers(api_key))
    response.raise_for_status()
    return response.json()


@app.get("/api/clips/{job_id}")
async def check_status(job_id: str):
    api_key = get_api_key()
    response = await _http.get(f"{REKA_BASE_URL}/{job_id}", headers=reka_headers(api_key))
    response.raise_for_status()
    return response.json()


app.mount("/static", StaticFiles(directory="static"), name="static")

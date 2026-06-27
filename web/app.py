"""
YT AI Clip - Web app backend (Gemini + yt-dlp + ffmpeg)
Run with: uvicorn app:app --reload
"""

import asyncio
import json
import os
import re
import subprocess
import tempfile
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import yt_dlp
from openai import OpenAI
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from moviepy import VideoFileClip, TextClip, CompositeVideoClip
from pydantic import BaseModel
from youtube_transcript_api import YouTubeTranscriptApi

CLIPS_DIR = Path("clips")
CLIPS_DIR.mkdir(exist_ok=True)

jobs: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(title="YT AI Clip", lifespan=lifespan)


class GenerateRequest(BaseModel):
    video_url: str
    prompt: str = "Create an engaging video clip highlighting the best moments"
    template: str = "moments"
    num_generations: int = 3
    min_duration: int = 0
    max_duration: int = 60
    subtitles: bool = True
    aspect_ratio: str = "9:16"
    resolution: int = 1080
    layout: str = "ai"


def get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY environment variable is not set.")
    return key


def extract_video_id(url: str) -> Optional[str]:
    match = re.search(r'(?:v=|/v/|youtu\.be/|/embed/)([a-zA-Z0-9_-]{11})', url)
    return match.group(1) if match else None


async def get_transcript(video_id: str) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, lambda: list(YouTubeTranscriptApi().fetch(video_id))
    )


def format_transcript(transcript: list) -> str:
    lines = []
    for entry in transcript:
        s = int(entry.start if hasattr(entry, "start") else entry["start"])
        text = entry.text if hasattr(entry, "text") else entry["text"]
        lines.append(f"[{s // 60:02d}:{s % 60:02d}] {text}")
    return "\n".join(lines)


FREE_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "meta-llama/llama-3.1-8b-instruct:free",
    "google/gemma-4-31b-it:free",
    "google/gemma-2-9b-it:free",
    "qwen/qwen3-14b:free",
    "qwen/qwen3-8b:free",
    "qwen/qwen3-30b-a3b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "mistralai/mistral-nemo:free",
    "mistralai/mistral-7b-instruct:free",
    "deepseek/deepseek-r1:free",
    "deepseek/deepseek-chat-v3-0324:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai/gpt-oss-120b:free",
    "google/gemma-4-26b-a4b-it:free",
]


async def ask_openrouter(api_key: str, transcript_text: str, prompt: str,
                         num: int, max_dur: int, min_dur: int) -> list:
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)

    user_prompt = f"""You are a viral video editor. Analyze this transcript and identify the {num} best moments for short clips.

Goal: {prompt}
Clip length: {min_dur}–{max_dur} seconds each

Return ONLY a JSON array of exactly {num} objects. No markdown, no explanation.
Each object must have:
- "title": catchy title (string)
- "caption": 1-2 sentence description (string)
- "hashtags": list of 3-5 hashtag strings
- "start_seconds": integer
- "end_seconds": integer

Transcript:
{transcript_text}"""

    loop = asyncio.get_event_loop()
    last_error = None

    for model in FREE_MODELS:
        try:
            response = await loop.run_in_executor(
                None,
                lambda m=model: client.chat.completions.create(
                    model=m,
                    messages=[{"role": "user", "content": user_prompt}],
                    temperature=0.7,
                ),
            )
            text = response.choices[0].message.content.strip()
            text = re.sub(r"^```(?:json)?\n?", "", text)
            text = re.sub(r"\n?```$", "", text)
            return json.loads(text)
        except Exception as e:
            last_error = e
            # On 429, wait the retry_after hint (or 10s) then try next model
            err_str = str(e)
            if "429" in err_str:
                wait = 10
                try:
                    import re as _re
                    m = _re.search(r"retry_after_seconds['\"]:\s*([\d.]+)", err_str)
                    if m:
                        wait = min(float(m.group(1)), 30)
                except Exception:
                    pass
                await asyncio.sleep(wait)
            continue

    raise RuntimeError(f"All free models rate-limited. Last error: {last_error}")


async def download_video(url: str, output_dir: str, resolution: int) -> str:
    ydl_opts = {
        "format": f"bestvideo[height<={resolution}][ext=mp4]+bestaudio[ext=m4a]/bestvideo[height<={resolution}]+bestaudio/best[height<={resolution}]/best",
        "outtmpl": os.path.join(output_dir, "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "merge_output_format": "mp4",
    }

    def _dl():
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            vid = info["id"]
            for ext in ("mp4", "mkv", "webm", "avi"):
                p = os.path.join(output_dir, f"{vid}.{ext}")
                if os.path.exists(p):
                    return p
            files = list(Path(output_dir).iterdir())
            if files:
                return str(files[0])
            raise RuntimeError("Downloaded video not found")

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _dl)


def _build_vf(aspect_ratio: str, resolution: int) -> str:
    ar_map = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1), "4:5": (4, 5)}
    w, h = ar_map.get(aspect_ratio, (9, 16))
    if w < h:
        return f"crop=ih*{w}/{h}:ih,scale=-2:{resolution}"
    return f"crop=iw:iw*{h}/{w},scale={resolution}:-2"


def _video_dims(aspect_ratio: str, resolution: int) -> tuple[int, int]:
    ar_map = {"9:16": (9, 16), "16:9": (16, 9), "1:1": (1, 1), "4:5": (4, 5)}
    w, h = ar_map.get(aspect_ratio, (9, 16))
    if w < h:
        vid_w = (resolution * w // h) & ~1  # ensure even
        return vid_w, resolution
    vid_h = (resolution * h // w) & ~1
    return resolution, vid_h


_FONTS_DIR = Path(__file__).parent / "fonts"

def _font_path() -> str:
    """Return path to Montserrat-Black.ttf, falling back to Helvetica Neue Bold."""
    bundled = _FONTS_DIR / "Montserrat-Black.ttf"
    if bundled.exists():
        return str(bundled)
    for d in ["/Library/Fonts", os.path.expanduser("~/Library/Fonts")]:
        for name in ("Montserrat-Black.ttf", "Montserrat-Bold.ttf"):
            p = os.path.join(d, name)
            if os.path.exists(p):
                return p
    return "/System/Library/Fonts/HelveticaNeue.ttc"


def chunk_transcript(transcript: list, clip_start: float, clip_end: float) -> list:
    """
    Split transcript into 2-3 word chunks timed to speech.
    Each chunk gets precise start/end times for rapid caption replacement.
    """
    chunks = []
    for entry in transcript:
        t_start = float(entry.start if hasattr(entry, "start") else entry["start"])
        t_dur   = float(entry.duration if hasattr(entry, "duration") else entry.get("duration", 2.0))
        text    = (entry.text if hasattr(entry, "text") else entry["text"]).strip()
        t_end   = t_start + t_dur

        if t_start >= clip_end or t_end <= clip_start:
            continue

        adj_start = max(0.0, t_start - clip_start)
        adj_end   = min(float(clip_end - clip_start), t_end - clip_start)

        words = text.upper().split()
        if not words:
            continue

        # 2 words for rapid engagement; 3 only when phrase is very short
        size = 2 if len(words) > 3 else 3
        groups = [words[i:i + size] for i in range(0, len(words), size)]
        seg_dur = adj_end - adj_start
        t_per = seg_dur / len(groups)

        for j, group in enumerate(groups):
            chunks.append({
                "words": group,
                "start": adj_start + j * t_per,
                "end":   adj_start + (j + 1) * t_per,
            })
    return chunks


def build_caption_clips(chunks: list, vid_w: int, vid_h: int, clip_duration: float) -> list:
    """
    Build MoviePy TextClip overlays — one per caption chunk.
    - Montserrat Black, fontsize=70, ALL CAPS
    - White body + yellow (#FFD700) last word, side-by-side on one line
    - 1.5px black stroke for legibility on any background
    - Centered horizontally, positioned at 78% down the frame
    - Instant cuts, no animation
    """
    font     = _font_path()
    FONTSIZE = 70
    WHITE    = "white"
    YELLOW   = "#FFD700"
    STROKE   = "black"
    SW       = 1.5  # stroke width — thin enough to be clean, thick enough to read
    y_pos    = int(vid_h * 0.78)  # 78% down the frame

    result = []
    for chunk in chunks:
        words   = chunk["words"]
        t_start = chunk["start"]
        t_end   = min(chunk["end"], clip_duration)
        dur     = max(0.05, t_end - t_start)

        def make_clip(text: str, color: str) -> TextClip:
            return TextClip(
                font=font, text=text, font_size=FONTSIZE,
                color=color, stroke_color=STROKE, stroke_width=SW,
                method="label",
            )

        if len(words) == 1:
            tc = (make_clip(words[0], YELLOW)
                  .with_position(("center", y_pos - FONTSIZE // 2))
                  .with_start(t_start).with_duration(dur))
            result.append(tc)
        else:
            body_text = " ".join(words[:-1]) + " "
            last_text = words[-1]
            body_tc   = make_clip(body_text, WHITE)
            last_tc   = make_clip(last_text, YELLOW)

            # Center the combined line
            total_w = body_tc.w + last_tc.w
            start_x = (vid_w - total_w) // 2
            text_y  = y_pos - body_tc.h // 2

            body_tc = (body_tc
                       .with_position((start_x, text_y))
                       .with_start(t_start).with_duration(dur))
            last_tc = (last_tc
                       .with_position((start_x + body_tc.w, text_y))
                       .with_start(t_start).with_duration(dur))
            result.extend([body_tc, last_tc])

    return result


async def cut_clip(input_path: str, output_path: str, start: int, end: int,
                   aspect_ratio: str, resolution: int):
    vf = _build_vf(aspect_ratio, resolution)
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start), "-i", input_path,
        "-t", str(end - start),
        "-vf", vf,
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]

    def _run():
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")

    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _run)


async def process_job(job_id: str, req: GenerateRequest, api_key: str):
    job = jobs[job_id]
    try:
        video_id = extract_video_id(req.video_url)
        if not video_id:
            raise ValueError("Could not extract YouTube video ID from URL")

        job["status"] = "fetching transcript"
        try:
            transcript = await get_transcript(video_id)
        except Exception as e:
            raise ValueError(f"Could not get transcript — make sure the video has captions enabled: {e}")

        job["status"] = "analyzing"
        moments = await ask_openrouter(
            api_key, format_transcript(transcript), req.prompt,
            req.num_generations, req.max_duration, req.min_duration,
        )

        job["status"] = "downloading"
        with tempfile.TemporaryDirectory() as tmpdir:
            video_file = await download_video(req.video_url, tmpdir, req.resolution)

            job["status"] = "rendering"
            output = []
            for i, moment in enumerate(moments):
                clip_name  = f"{job_id}_{i}.mp4"
                clip_path  = str(CLIPS_DIR / clip_name)
                clip_start = int(moment["start_seconds"])
                clip_end   = int(moment["end_seconds"])

                await cut_clip(
                    video_file, clip_path,
                    clip_start, clip_end,
                    req.aspect_ratio, req.resolution,
                )
                output.append({
                    "title": moment.get("title", f"Clip {i + 1}"),
                    "caption": moment.get("caption", ""),
                    "hashtags": moment.get("hashtags", []),
                    "video_url": f"/clips/{clip_name}",
                })

        job["status"] = "completed"
        job["output"] = output

    except Exception as exc:
        job["status"] = "failed"
        job["error_message"] = str(exc)


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    api_key = get_api_key()
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"id": job_id, "status": "queued"}
    asyncio.create_task(process_job(job_id, req, api_key))

    async def event_stream():
        seen: set[str] = set()
        while True:
            job = jobs.get(job_id, {})
            status = job.get("status", "queued")
            if status not in seen:
                seen.add(status)
                evt: dict = {"id": job_id, "status": status}
                if status == "completed":
                    evt["output"] = job.get("output", [])
                if status == "failed":
                    evt["error_message"] = job.get("error_message", "Unknown error")
                yield f"data: {json.dumps(evt)}\n\n"
            if status in ("completed", "failed"):
                break
            await asyncio.sleep(1)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/clips")
async def list_clips_api():
    return {"jobs": list(jobs.values())}


@app.get("/api/clips/{job_id}")
async def check_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


app.mount("/clips", StaticFiles(directory="clips"), name="clips")
app.mount("/static", StaticFiles(directory="static"), name="static")

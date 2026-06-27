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
    "google/gemma-4-31b-it:free",
    "google/gemma-4-26b-a4b-it:free",
    "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai/gpt-oss-120b:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
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


def _font_name() -> str:
    """Use Montserrat if installed, fall back to Helvetica Neue (always on macOS)."""
    font_dirs = ["/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    targets = ("Montserrat-Bold.ttf", "Montserrat-Black.ttf", "Montserrat Bold.ttf")
    for d in font_dirs:
        for name in targets:
            if os.path.exists(os.path.join(d, name)):
                return "Montserrat"
    return "Helvetica Neue"


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

        size = 3
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


def generate_premium_ass(chunks: list, vid_w: int, vid_h: int) -> str:
    """
    Build an ASS subtitle file styled after Iman Gadzhi's premium caption aesthetic:
      - Montserrat Black / Helvetica Neue Bold, ALL CAPS
      - Tight letter spacing (-1)
      - Pure white body text, soft yellow (#FFD700) on the last word per chunk
      - Soft drop shadow only — NO outline/stroke (BorderStyle=1, Outline=0, Shadow=3)
      - Shadow color: black @ 50% opacity
      - Position: center-bottom at ~80% down the frame
      - Clean cuts — no animations
    """
    font      = _font_name()
    WHITE     = "&H00FFFFFF"   # #FFFFFF  — primary text
    YELLOW    = "&H0000D7FF"   # #FFD700 in ASS BGR order
    SHADOW_C  = "&H80000000"   # black @ 50% alpha — soft, not cheap
    font_size = 12
    margin_v  = int(vid_h * 0.18)             # 18% from bottom ≈ 82% down

    header = f"""\
[Script Info]
ScriptType: v4.00+
PlayResX: {vid_w}
PlayResY: {vid_h}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{font_size},{WHITE},&H000000FF,&H00000000,{SHADOW_C},-1,0,0,0,100,100,-1,0,1,0,3,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def fmt(s: float) -> str:
        h = int(s // 3600)
        m = int((s % 3600) // 60)
        sec = s % 60
        return f"{h}:{m:02d}:{sec:05.2f}"

    lines = []
    for chunk in chunks:
        words   = chunk["words"]
        t_start = chunk["start"]
        t_end   = chunk["end"]

        if len(words) == 1:
            # Solo word → yellow (stands alone as emphasis)
            text = f"{{\\c{YELLOW}\\shad3\\4c{SHADOW_C}}}{words[0]}"
        else:
            # Body words in white, last word in yellow for emphasis
            body = " ".join(words[:-1])
            last = words[-1]
            text = (
                f"{{\\c{WHITE}\\shad3\\4c{SHADOW_C}}}{body} "
                f"{{\\c{YELLOW}}}{last}{{\\c{WHITE}}}"
            )

        lines.append(f"Dialogue: 0,{fmt(t_start)},{fmt(t_end)},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


async def cut_clip(input_path: str, output_path: str, start: int, end: int,
                   aspect_ratio: str, resolution: int, ass_content: str = ""):
    ass_path = output_path + ".ass"
    vf = _build_vf(aspect_ratio, resolution)

    if ass_content:
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write(ass_content)
        escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:")
        vf += f",ass={escaped}"

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
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed: {result.stderr[-500:]}")
        finally:
            if os.path.exists(ass_path):
                os.remove(ass_path)

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
            vid_w, vid_h = _video_dims(req.aspect_ratio, req.resolution)
            output = []
            for i, moment in enumerate(moments):
                clip_name  = f"{job_id}_{i}.mp4"
                clip_path  = str(CLIPS_DIR / clip_name)
                clip_start = int(moment["start_seconds"])
                clip_end   = int(moment["end_seconds"])

                if req.subtitles:
                    chunks = chunk_transcript(transcript, clip_start, clip_end)
                    ass = generate_premium_ass(chunks, vid_w, vid_h)
                else:
                    ass = ""

                await cut_clip(
                    video_file, clip_path,
                    clip_start, clip_end,
                    req.aspect_ratio, req.resolution, ass,
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

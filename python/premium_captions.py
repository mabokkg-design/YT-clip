"""
premium_captions.py — Iman Gadzhi-style premium caption generator
Python + ffmpeg (no MoviePy needed)

Design spec:
  - Font:     Montserrat Black / Helvetica Neue Bold  |  ALL CAPS  |  tight spacing
  - Colors:   White body  +  Yellow (#FFD700) emphasis on last word per chunk
  - Shadow:   Soft black drop shadow (offset 2px, opacity 60%) — NO outline
  - Position: Center-bottom at ~80% down the frame
  - Chunking: 2 words at a time, timed exactly to speech
  - Animation: instant cut (no pop-in, no bounce)

Usage:
    python premium_captions.py --video input.mp4 --transcript transcript.json --out output.mp4

    Transcript JSON format (from youtube-transcript-api or manual):
    [
      {"text": "Never gonna give you up", "start": 43.2, "duration": 1.8},
      ...
    ]
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# 1.  FONT — Montserrat if installed, Helvetica Neue as macOS fallback
# ---------------------------------------------------------------------------

def _detect_font() -> str:
    font_dirs = ["/Library/Fonts", os.path.expanduser("~/Library/Fonts")]
    candidates = ("Montserrat-Bold.ttf", "Montserrat-Black.ttf", "Montserrat Bold.ttf")
    for d in font_dirs:
        for name in candidates:
            if os.path.exists(os.path.join(d, name)):
                return "Montserrat"
    # Helvetica Neue is always present on macOS
    return "Helvetica Neue"


# ---------------------------------------------------------------------------
# 2.  CHUNKING — split transcript into 2-word blocks timed to speech
# ---------------------------------------------------------------------------

def chunk_transcript(transcript: list[dict], clip_start: float = 0.0,
                     clip_end: float = float("inf"), words_per_chunk: int = 2) -> list[dict]:
    """
    Break transcript entries into rapid 2-word caption chunks.

    Each returned dict:
        {"words": ["NEVER", "GONNA"], "start": 0.0, "end": 0.55}

    Args:
        transcript:      List of {text, start, duration} dicts.
        clip_start:      Offset (seconds) of clip start in the source video.
        clip_end:        Offset (seconds) of clip end in the source video.
        words_per_chunk: How many words per screen (2 = fastest / most engaging).
    """
    chunks = []

    for entry in transcript:
        t_start = float(entry.get("start", 0))
        t_dur   = float(entry.get("duration", 2.0))
        text    = str(entry.get("text", "")).strip()
        t_end   = t_start + t_dur

        # Skip entries outside the clip window
        if t_start >= clip_end or t_end <= clip_start:
            continue

        # Adjust timestamps to be relative to clip start
        adj_start = max(0.0, t_start - clip_start)
        adj_end   = min(clip_end - clip_start, t_end - clip_start)

        # ALL CAPS as per spec
        words = text.upper().split()
        if not words:
            continue

        # Divide the entry's duration evenly across word groups
        groups   = [words[i:i + words_per_chunk] for i in range(0, len(words), words_per_chunk)]
        seg_dur  = adj_end - adj_start
        t_per    = seg_dur / len(groups)

        for j, group in enumerate(groups):
            chunks.append({
                "words": group,
                "start": adj_start + j * t_per,
                "end":   adj_start + (j + 1) * t_per,
            })

    return chunks


# ---------------------------------------------------------------------------
# 3.  ASS GENERATOR — full premium styling as Advanced SubStation Alpha
# ---------------------------------------------------------------------------

def generate_premium_ass(chunks: list[dict], vid_w: int, vid_h: int) -> str:
    """
    Build an ASS subtitle string with Iman Gadzhi-style premium captions.

    Design decisions mapped to ASS parameters:
        BorderStyle=1, Outline=0, Shadow=3   → shadow only, no stroke
        BackColour=&H99000000                → black shadow @ 60% opacity
        Spacing=-1                           → tight letter spacing
        Bold=-1                              → bold weight
        Alignment=2                          → center-bottom
        MarginV = vid_h * 0.18               → sits at ~82% down the frame
        \\c&H0000D7FF&                        → yellow (#FFD700 in ASS BGR order)
        \\shad3\\4c&H99000000&               → per-line shadow override
    """

    font     = _detect_font()
    WHITE    = "&H00FFFFFF"   # #FFFFFF — primary text
    YELLOW   = "&H0000D7FF"   # #FFD700 in ASS BGR (Blue=00, Green=D7, Red=FF)
    SHADOW_C = "&H99000000"   # black @ 60% opacity  (0x99 = 153 ≈ 60% of 255)

    # Scale font size relative to frame height (~5.5% feels premium at any res)
    font_size = max(36, int(vid_h * 0.055))
    # 18% margin from bottom places text at ~82% down — sweet spot for shorts
    margin_v  = int(vid_h * 0.18)

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

    def _fmt(s: float) -> str:
        """Format seconds → ASS timestamp  H:MM:SS.cc"""
        h   = int(s // 3600)
        m   = int((s % 3600) // 60)
        sec = s % 60
        return f"{h}:{m:02d}:{sec:05.2f}"

    lines = []
    for chunk in chunks:
        words   = chunk["words"]
        t_start = chunk["start"]
        t_end   = chunk["end"]

        if len(words) == 1:
            # Single word alone = key moment → render fully in yellow
            text = f"{{\\c{YELLOW}\\shad3\\4c{SHADOW_C}}}{words[0]}"
        else:
            # 2+ words: body in white, LAST word in yellow (the emphasis word)
            body = " ".join(words[:-1])
            last = words[-1]
            text = (
                f"{{\\c{WHITE}\\shad3\\4c{SHADOW_C}}}{body} "
                f"{{\\c{YELLOW}}}{last}{{\\c{WHITE}}}"
            )

        lines.append(f"Dialogue: 0,{_fmt(t_start)},{_fmt(t_end)},Default,,0,0,0,,{text}")

    return header + "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 4.  VIDEO PROCESSING — ffmpeg burn-in
# ---------------------------------------------------------------------------

def get_video_dimensions(video_path: str) -> tuple[int, int]:
    """Return (width, height) of video using ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    w, h = result.stdout.strip().split(",")
    return int(w), int(h)


def burn_captions(video_path: str, transcript: list[dict], output_path: str,
                  clip_start: float = 0.0, clip_end: float = None,
                  words_per_chunk: int = 2) -> None:
    """
    Apply premium captions to a video and save to output_path.

    Args:
        video_path:      Path to source video.
        transcript:      List of {text, start, duration} dicts.
        output_path:     Where to write the output video.
        clip_start:      Start offset if video is already trimmed to a clip (default 0).
        clip_end:        End offset. Defaults to the full video duration.
        words_per_chunk: Words per caption block (2 = max engagement).
    """
    vid_w, vid_h = get_video_dimensions(video_path)

    if clip_end is None:
        # Probe duration
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        clip_end = float(result.stdout.strip())

    # Build caption chunks and ASS file
    chunks = chunk_transcript(transcript, clip_start, clip_end, words_per_chunk)
    ass_content = generate_premium_ass(chunks, vid_w, vid_h)

    with tempfile.NamedTemporaryFile(suffix=".ass", delete=False, mode="w", encoding="utf-8") as f:
        f.write(ass_content)
        ass_path = f.name

    try:
        # Escape path for ffmpeg vf filter (colons break the filter graph)
        escaped = ass_path.replace("\\", "\\\\").replace(":", "\\:")
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-vf", f"ass={escaped}",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy",
            output_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr[-800:]}")
        print(f"✓ Saved: {output_path}")
    finally:
        os.remove(ass_path)


# ---------------------------------------------------------------------------
# 5.  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Add Iman Gadzhi-style captions to a video.")
    parser.add_argument("--video",      required=True,  help="Input video file")
    parser.add_argument("--transcript", required=True,  help="Transcript JSON file")
    parser.add_argument("--out",        required=True,  help="Output video file")
    parser.add_argument("--start",      type=float, default=0.0,  help="Clip start offset in seconds")
    parser.add_argument("--end",        type=float, default=None, help="Clip end offset in seconds")
    parser.add_argument("--words",      type=int,   default=2,    help="Words per caption chunk (default 2)")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        sys.exit(f"Error: video not found: {args.video}")
    if not os.path.exists(args.transcript):
        sys.exit(f"Error: transcript not found: {args.transcript}")

    with open(args.transcript, encoding="utf-8") as f:
        transcript = json.load(f)

    burn_captions(
        video_path=args.video,
        transcript=transcript,
        output_path=args.out,
        clip_start=args.start,
        clip_end=args.end,
        words_per_chunk=args.words,
    )


if __name__ == "__main__":
    main()

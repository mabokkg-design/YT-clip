# YT AI Clip — Web App

A browser-based UI for generating short clips from YouTube videos using the Reka AI API.

## Setup

**Requirements**: Python 3.9+

```bash
cd web
pip install -r requirements.txt
export REKA_API_KEY=your_key_here
uvicorn app:app --reload
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Features

- Paste a YouTube URL and generate a clip in real-time
- Streams status updates live (queued → downloading → indexing → processing → done)
- Saves the Job ID so you can recover if the page is refreshed
- Configurable options: template, aspect ratio, duration, prompt
- Download or copy the clip URL when ready
- Check the status of any existing job by ID

## Options

| Option | Values | Default |
|---|---|---|
| Template | `moments`, `compilation` | `moments` |
| Aspect ratio | `9:16`, `16:9`, `1:1`, `4:5` | `9:16` |
| Min duration | seconds | `0` |
| Max duration | seconds | `30` |
| Prompt | free text | `Create an engaging video clip…` |

## Links

- [Reka API Docs](https://docs.reka.ai/vision/highlight-reel-generation)
- [Get a free API key](https://link.reka.ai/free)
- [Reka Discord](https://link.reka.ai/discord)

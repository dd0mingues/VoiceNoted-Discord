# 🛠️ VoiceNoted-Discord: Technical Architecture & Integration Specification

This document provides an in-depth technical analysis of the architectural design, API integrations, memory models, audio pipeline, and cloud runtime mechanisms implemented in **VoiceNoted-Discord**.

---

## 📐 High-Level Architecture Overview

```
                        ┌─────────────────────────────────────────────────────────────┐
                        │                       Discord Cloud                         │
                        └───────────────────────┬─────────────────────────────────────┘
                                                │
                          Discord Gateway v10   │  Inbound Event: MESSAGE_CREATE
                         (WebSocket Persistent) │  - message.flags & 8192 (IS_VOICE_MESSAGE)
                                                ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│ VoiceNoted-Discord Engine (bot.py)                                                          │
│                                                                                             │
│  ┌──────────────────────┐        ┌───────────────────────┐        ┌──────────────────────┐  │
│  │ VoiceNotedClient     │        │ In-Memory Pipeline    │        │ Message Dispatcher   │  │
│  │ (discord.Client)     ├───────►│ - attachment.read()   ├───────►│ - split_text(1900)   │  │
│  │                      │        │ - In-memory bytes     │        │ - message.reply()    │  │
│  │ - voice_states=False │        │ - Size Guard (≤25MB)  │        │ - message.channel()  │  │
│  └──────────┬───────────┘        └───────────┬───────────┘        └──────────────────────┘  │
│             │                                │                                              │
│             ▼                                ▼                                              │
│  ┌──────────────────────┐        ┌───────────────────────┐                                  │
│  │ Embedded HTTP Server │        │ AsyncGroq LPU Client  │                                  │
│  │ (aiohttp.web)        │        │ - whisper-large-v3    │                                  │
│  │ - Port: $PORT        │        │ - language="pt"       │                                  │
│  │ - Routes: /, /health │        │ - temperature=0.0     │                                  │
│  │ - 200 OK Keep-Alive  │        │ - WoW / PT-PT Prompt  │                                  │
│  └──────────────────────┘        └───────────┬───────────┘                                  │
└──────────────────────────────────────────────┼──────────────────────────────────────────────┘
                                               │
                           HTTPS (TLS 1.3)     │  Multi-Part Form Stream
                           Groq Cloud API      │  (No disk files created)
                                               ▼
                        ┌─────────────────────────────────────────────────────────────┐
                        │                   Groq LPU Cloud Platform                   │
                        │                 Whisper Large-v3 Inference                  │
                        └─────────────────────────────────────────────────────────────┘
```

---

## 1. Discord Gateway & Event Lifecycle

### 1.1 Discord Gateway Protocol (v10)
- **Persistent WebSocket**: The application initializes an asynchronous persistent duplex connection to `wss://gateway.discord.gg/?v=10&encoding=json`.
- **Intents Matrix**:
  ```python
  intents = discord.Intents.default()
  intents.message_content = True  # Privileged Intent (1 << 15)
  intents.messages = True         # Receive message events in guilds
  intents.voice_states = False    # Disables voice state tracking
  ```
  - `message_content = True` is required by Discord's Privileged Gateway policy introduced in August 2022. Without this intent, Discord redacts message attachments, flags, and text bodies from `MESSAGE_CREATE` packets.
  - `voice_states = False`: Explicitly avoids receiving `VOICE_STATE_UPDATE` packets. The application deliberately lacks audio gateway dependencies (`PyNaCl` / `davey`), ensuring it is technically incapable of joining voice channels, intercepting WebRTC audio streams, or recording live calls.

### 1.2 Voice Message & Audio Detection Mechanism
Voice messages in Discord are not transmitted via special distinct events; they arrive as standard `MESSAGE_CREATE` events with distinct metadata:
```python
VOICE_MESSAGE_FLAG = 8192  # Bit 13: (1 << 13) IS_VOICE_MESSAGE
```
The detection pipeline executes across three ingestion vectors:
1. **Bot Filter**: `if message.author.bot: return` prevents recursive bot loops and self-triggering.
2. **Direct Attachments & Voice Flags**: Evaluates `bool(message.flags.value & 8192)`. If true, the message is a native Discord voice message. Evaluates `content_type` (`audio/*`) and extensions (`.ogg`, `.mp3`, `.wav`, `.m4a`).
3. **Forwarded Messages (Discord `message_snapshots`)**: Discord's Message Forwarding feature does not populate top-level `message.attachments`. Instead, it embeds `message.message_snapshots`, containing snapshots of the original message and its attachments. The pipeline traverses `snapshot.attachments` and checks `snapshot.flags` for bit `8192`.
4. **Referenced Messages (Replies)**: If a user replies to an existing voice message with a mention, `message.reference.resolved` is queried for historical audio payloads.


---

## 2. In-Memory Streaming Pipeline (Zero Disk I/O)

Traditional bots download audio files to temporary disk storage (`/tmp/voice.ogg`), process them, and invoke cleanup handlers. This creates critical liabilities in cloud/container environments:
- Container disk exhaustion and inode leaks if cleanup hooks fail on SIGTERM.
- Flash wear on SSD-based virtual volumes.
- Additional file I/O latency.

### 2.1 The Zero-Disk Buffer Architecture
```python
async with message.channel.typing():
    audio_bytes = await attachment.read()
    transcription = await groq_client.audio.transcriptions.create(
        file=(filename, audio_bytes, content_type),
        model="whisper-large-v3",
        language="pt",
        temperature=0.0,
        prompt=TRANSCRIPTION_PROMPT,
    )
```
- **Asynchronous Chunked Download**: `attachment.read()` fetches the Discord CDN audio stream directly into RAM as an immutable `bytes` buffer.
- **Multipart Tuple Streaming**: Groq's SDK (backed by `httpx.AsyncClient`) accepts in-memory multipart tuples in the format `(filename, bytes_data, content_type)`. This is serialized directly into HTTP `multipart/form-data` payload buffers and streamed over TLS.
- **Preemptive Size Guard**: The 25MB boundary enforced by the Groq Whisper API is validated twice:
  1. Header check: `attachment.size > 26_214_400` (aborts download before bandwidth is wasted).
  2. Buffer check: `len(audio_bytes) > 26_214_400` (prevents oversized payloads if Content-Length was missing or falsified).

---

## 3. Groq LPU Acceleration & AI Prompt Conditioning

### 3.1 Inference Engine: Groq LPU vs. Standard GPUs
- **Low Latency**: Traditional cloud GPUs (Nvidia A10G/T4) process Whisper Large-v3 with latency scaling roughly $10\times$ to $20\times$ slower than real-time. Groq's Language Processing Unit (LPU) architecture operates on deterministic tensor streaming, processing a 30-second voice note in **~0.4 to 1.2 seconds**.
- **Endpoint Specification**: Calls exclusively `groq_client.audio.transcriptions.create` (and never `/audio/translations`).

### 3.2 Prompt Conditioning for PT-PT and Gaming / WoW
OpenAI's Whisper architecture is heavily biased toward American English and Brazilian Portuguese (PT-BR) due to dataset distribution. Without conditioning:
- English gaming terms pronounced with Portuguese accents are mangled:
  - *"Guild"* $\rightarrow$ *"Gil de"* or *"Gilda"*
  - *"Raid"* $\rightarrow$ *"Reib"* or *"Rede"*
  - *"Loot"* $\rightarrow$ *"Luto"* (mourning)
  - *"Wipe"* $\rightarrow$ *"Uaip"*
- European Portuguese phonetics are misidentified as Brazilian phonetics (*"estou a fazer"* corrupted to *"estou fazendo"*).

To counteract this, the prompt injects explicit phonetic and semantic anchors:
```python
TRANSCRIPTION_PROMPT = (
    "Transcrição em português de Portugal (PT-PT) de uma conversa entre gamers sobre World of Warcraft (WoW). "
    "Manter termos em inglês quando ditos: Guild, Guild Master, GM, Raid, Raid Leader, RL, PUG, pugar, "
    "loot, loot drama, wipe, boss, aggro, pull, tank, healer, DPS, mythic, M+, dungeon, addon, Discord. "
    "Calão português: gajos, malta, cenas, fixe, caralho, caraças, acumulado."
)
```

### 3.3 Anti-Hallucination & Temperature Tuning
- **`temperature=0.0`**: Whisper incorporates random sampling at temperatures $> 0$. Setting temperature to zero enforces greedy deterministic decoding, reducing hallucinations during silent breath pauses or low-bitrate microphone noise.
- **Silence Artifact Suppression**: Whisper models frequently output spurious subtitles on dead silence (e.g., `"Obrigado."`, `"Legendas pela comunidade..."`). The code intercepts and matches the raw text against `HALLUCINATION_PHRASES`, suppressing empty or hallucinated responses completely.

---

## 4. Discord Message Chunking Algorithm

Discord rejects any single message exceeding **2,000 Unicode characters** (`400 Bad Request: HTTPException 50035`). A 4-minute voice note regularly generates 3,000–5,000 characters of transcribed text.

### 4.1 Boundary-Aware Windowing (`split_text`)
Instead of naively chopping strings at character index 1900 (which cuts words or URLs in half), `split_text()` uses priority-based window slicing:

```python
def split_text(text: str, max_length: int = 1900) -> list[str]:
```
For each window of length `max_length`:
1. **Priority 1 (Sentence Delimiters)**: Looks backward from index 1900 using `rfind()` for `\n`, `. `, `! `, or `? `. Slices cleanly after the punctuation mark.
2. **Priority 2 (Word Delimiters)**: If no sentence terminal exists in the segment, searches backward for the last whitespace `' '`.
3. **Priority 3 (Fallback)**: If a continuous string with no whitespace exceeds 1900 characters, executes a hard cut at `max_length`.

### 4.2 Reply Chaining
- Chunk 1 is formatted with `🎙️ **Transcrição:**\n` and sent via `await message.reply(..., mention_author=False)`.
- Chunks $2 \dots N$ are dispatched sequentially via `await message.channel.send(...)` into the same channel context.

---

## 5. Cloud Runtime Architecture (Render / Koyeb)

### 5.1 Dual Lifecycle Model
Discord bots operate as outbound WebSocket consumers and do not inherently expose listening TCP ports. However, cloud Platforms-as-a-Service (Render, Koyeb, Railway) manage application health via inbound HTTP port monitoring. If an app does not bind to `$PORT` within ~120 seconds, the orchestrator terminates the container.

### 5.2 Embedded Asynchronous Health-Check Server
We solve this by subclassing `discord.Client` and implementing `setup_hook`:
```python
class VoiceNotedClient(discord.Client):
    async def setup_hook(self) -> None:
        port = os.getenv("PORT")
        if port:
            from aiohttp import web
            async def handle_health(request: web.Request) -> web.Response:
                return web.Response(text="VoiceNoted Discord Bot is active!\nStatus: 200 OK")

            app = web.Application()
            app.router.add_get("/", handle_health)
            app.router.add_get("/health", handle_health)
            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, "0.0.0.0", int(port))
            await site.start()
```
- **Same Event Loop**: Uses `aiohttp.web` running on Python's primary `asyncio` event loop. It introduces **zero thread overhead**.
- **Conditional Binding**: Only starts when the cloud platform injects the `PORT` environment variable. When running locally on desktop/bare-metal, it remains inert.

### 5.3 Render Free Tier Keep-Alive Architecture
```
┌─────────────────┐  HTTP GET every 5m   ┌───────────────────────┐  Duplex WS   ┌───────────────┐
│  UptimeRobot    ├─────────────────────►│  Render Web Service   ├─────────────►│ Discord       │
│  (Free Monitor) │                      │  (Port 10000 /health) │              │ Gateway v10   │
└─────────────────┘                      └───────────────────────┘              └───────────────┘
```
1. Render's free tier idles containers after 15 minutes of zero inbound HTTP traffic.
2. UptimeRobot sends an HTTP GET request to `https://<service-name>.onrender.com/health` every 5 minutes.
3. The embedded `aiohttp.web` server returns `200 OK` in <50ms.
4. Render's traffic analyzer resets the 15-minute inactivity timer, allowing the underlying Discord WebSocket to run uninterrupted 24/7 at $0 cost.

---

## 6. Containerization & Production Specifications

### 6.1 Docker Container Strategy
The project uses `python:3.11-slim`:
- **Base Footprint**: Minimal Debian slim base image (~130MB).
- **Environment Flags**:
  - `PYTHONDONTWRITEBYTECODE=1`: Prevents writing `.pyc` files during execution.
  - `PYTHONUNBUFFERED=1`: Ensures stdout/stderr logs are flushed immediately to container log collectors (Render logs / `docker logs`).
- **Layer Caching**: `requirements.txt` is copied and installed prior to copying `bot.py`, ensuring rebuilds when code changes complete in under 2 seconds.

### 6.2 Security Best Practices
- **Token Segregation**: Tokens are injected exclusively via runtime environment variables (`.env` locally, encrypted secrets on Render/Koyeb).
- **Git Hardening**: `.gitignore` strictly rejects `.env`, `.env.*`, `__pycache__`, and virtual environments.
- **Fail-Fast Boot Validation**: `validate_environment()` executes before any network connection is attempted, preventing infinite reconnect loops if credentials are absent.

# ChatHippo — Local AI Chat for Ollama

Private, fast, and local-first chat interface for running LLMs with Ollama.

Quick Download: https://github.com/ck4445/ChatHippo/releases


## Overview

ChatHippo is a lightweight, single-machine chat UI that feels familiar (ChatGPT-like) while keeping your data local. It connects to your local Ollama server, streams responses, persists chat history on disk, and adds a few quality-of-life features like a context meter, memories, personalities, attachments, and a curated Model Store.


## Key Features

- Chat with local models: Lists installed Ollama models, lets you pick per-chat, and shows a tps meter while streaming.
- Chat history: Stores conversations locally as JSON, with rename and delete. Supports “regenerate last response”.
- Memories: Persistent, optional working memory. Auto-saves facts when models emit <add_to_memory>…</add_to_memory> tags; includes manual add/edit/delete and enable/disable.
- Personalities: Blend up to 3 built‑in personalities (e.g., Friendly, Professional, Nerd) to influence tone and style.
- Language + verbosity: Choose preferred response language and verbosity target.
- Context meter: Estimates tokens used by system + history + current draft (client-side) against a 32K budget from the server.
- Attachments: Attach plain‑text files to the next message; token‑aware limits per file and per draft; shows total token budget.
- Voice input (Chrome): Dictate prompts using the Web Speech API.
- Themes + animations: Dark/light theme and toggleable loader animations.
- Model Store: Browse a curated list (families, variants, specs), install via Ollama pull with live progress and cancellable jobs.


## Architecture

- Backend: Python Flask app served by Waitress. Static SPA (HTML/CSS/JS) is served by Flask, including a catch‑all route for client‑side navigation.
- Frontend: Vanilla JS + `marked` + `highlight.js` + KaTeX for Markdown/code/math. Voice via `webkitSpeechRecognition` when available.
- Data: Local folder `data/` for user profile, settings, chats, drafts/uploads, memories, and model label metadata.
- Ollama: All inference and model management route to a local Ollama server (default `http://127.0.0.1:11434`, overridable via `OLLAMA_HOST`).


## Data Layout

- `data/user.json`: Display name and plan (local).
- `data/settings.json`: UI theme/animation, default model, personality selection, language, verbosity, developer flags (e.g., disable system prompt), and context meter setting.
- `data/chats/*.json`: One JSON per chat with messages and metadata.
- `data/uploads/<chat_id>/drafts/<draft_id>/`: Temporary attachment files and descriptors while drafting a message.
- `data/memories.json`: Global memory store; capped count and length with “enabled” toggle.
- `data/modelnames.json`: Optional map of model IDs to friendly labels and descriptions for the UI.
- `data/new_models.json`: Optional “aliases” you define (display name → base model + alias‑specific system prompt).


## API Surface (for contributors)

- Models: `GET /api/models` (with friendly meta), `POST /api/abort` (cancel stream by sid).
- Chat: `POST /api/chat/stream` (NDJSON stream), CRUD under `/api/chats`, full export `GET /api/chats/all`.
- Attachments (text only): `GET/POST/DELETE /api/chats/<cid>/drafts/<did>/attachments[/<attid>]` with token budgeting.
- Settings & user: `GET/POST /api/settings`, `GET/POST /api/user`.
- Memories: `GET/POST /api/memories`, `PUT/DELETE /api/memories/<id>`, `POST /api/memories/toggle`.
- Model Store: `GET /api/store/models`, `POST /api/store/pull`, `GET /api/store/jobs`, `POST /api/store/jobs/<id>/cancel`, `GET /api/store/jobs/<id>/stream`.


## Installation

Prereqs:
- Python 3.10+
- Ollama installed and running (`ollama serve`).

Install (dev):
```bash
git clone https://github.com/ck4445/ChatHippo.git
cd ChatHippo
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

pip install flask waitress requests
```

Configure Ollama (optional if running on default):
```bash
export OLLAMA_HOST="http://127.0.0.1:11434"    # macOS/Linux
$env:OLLAMA_HOST = "http://127.0.0.1:11434"   # Windows PowerShell
```

Run:
```bash
python app.py
# or production-ish
waitress-serve --listen=0.0.0.0:8080 app:app
```

Windows helper: `start.bat` clears port 8080 and runs `python app.py`.


## Usage Notes

- Pick model: Click “Models” and choose any installed model. Use the Model Store to install more.
- Draft + send: Compose on Home or in Chat. Press Enter to send (Shift+Enter for newline). Voice input on Chrome.
- Abort generation: Click the stop button while streaming.
- Regenerate last: Use the “Reload” tool on the last assistant message to resend the prior user message.
- Attachments: Use the paperclip in Chat or drag text files onto the composer. Token limits: 20k per file / 25k per draft. Only plain‑text types; images are not attached for vision models.
- Memories: Toggle in the Memories modal. Add/edit/delete manually, or let the model write `<add_to_memory>fact</add_to_memory>` to save facts automatically.
- Settings: Theme, animations, default model, personalities (max 3), language, verbosity, context meter, and a developer switch to disable the system prompt entirely.
- Search: Search chats and messages from the sidebar Search.


## Model Store

The Model Store page groups a curated Ollama catalog from `list.txt` by families and variants:
- Filtering: All, Multimodal, Reasoning, Embeddings, Assistant, Installed.
- Recommended variant: Picks a small, reasonable default per family.
- Install flow: Queues a pull from Ollama and streams progress per job; cancel supported.
- Details: Per‑variant specs and “Copy ID” for quick usage.


## Security & Privacy

- Local-first: All inference is via your local Ollama instance. No chat data is sent off-machine by this app.
- Attachments: File contents you attach are injected into the model context. Avoid sensitive materials unless you fully trust your local environment.
- Memory: Automatic facts are saved only when memories are enabled; entries are visible and editable.
- Disclaimer: Portions of this project were generated with AI. Review code before running if you have security constraints.


## Roadmap

- Vision attachments: Image input pipeline for supported multimodal models.
- RAG helpers: Local document collections and retrieval.
- Rich prompts: Per-chat system prompt editor and presets.
- Functions/tools: Surface model tool/function calling when supported by Ollama builds.
- Import/export: Zip chats, attachments, and settings.
- Packaging: Streamlined cross‑platform bundles and auto‑updates.


## Contributing

- Issues: Use GitHub Issues for bugs and feature requests.
- PRs: Keep diffs focused and documented. Describe UX and API changes clearly.
- Style: Maintain the minimal, framework‑free frontend and Flask API patterns.


## License

No license has been provided yet. If you intend to fork/use, please open an issue or add a license (e.g., MIT) to clarify terms.


## Acknowledgements

- Ollama for local model serving.
- marked, highlight.js, KaTeX for rendering.
- Community model authors listed in `list.txt`.

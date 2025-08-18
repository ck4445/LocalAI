# Chathippo

A private, fast, local AI chat interface powered by Ollama.

## Short description

Chathippo is a lightweight chat UI that mimics the familiar ChatGPT experience.  
It connects to local Ollama models and keeps all data on the user machine.

## Disclaimer

This project was largely generated with AI. That can introduce code or design issues.  
Because Chathippo runs locally and relies on local models it reduces remote exposure.  
Local execution does not eliminate all risk. Review the code before running. Use caution with sensitive data.

Quick Download: https://github.com/ck4445/ChatHippo/releases

## Features

- Chat with any local Ollama model.
- Local chat history saved on disk.
- Support for system prompts and personalities.
- Clean, minimal UI with light and dark themes.
- Browse a simple model "store".
- Context bar for quick reference.

## Intended audience

Designed for casual users who want a fast, familiar chat interface.  
If you need fine-grained control for production or research use the official Ollama UI or LM Studio.

## Roadmap

Planned items, not guaranteed:
- `llama.cpp` support.
- More built-in personalities.
- A working memory system.

## Download

Get Windows builds from the GitHub Releases page:  
`https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME/releases/latest`

## Run from source (developers)

1. Clone the repo:
```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
````

2. Create and activate a virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

3. Create `requirements.txt` or use the example below and install dependencies:

```txt
flask
waitress
requests
python-dotenv
```

```bash
pip install -r requirements.txt
```

4. Configure Ollama access. Install and run Ollama separately according to its docs.
   Optionally set an environment variable if your app uses one:

```bash
export OLLAMA_HOST="http://localhost:11434"
# Windows (PowerShell)
$env:OLLAMA_HOST="http://localhost:11434"
```

5. Run the app:

```bash
python app.py
# or, for production
waitress-serve --listen=0.0.0.0:8080 app:app
```

## Config and data

* Chat history is stored locally in the repo's data folder by default.
* Review `config.example.env` or the app's config for paths and options.

## Troubleshooting

* If the UI cannot reach models check that Ollama is running and reachable.
* If dependencies fail, recreate the virtual environment and reinstall.
* Inspect logs printed to the console for crash details.

## Contributing

Contributions are welcome. Open an issue or submit a pull request. Keep changes small and documented.

## License

Add a license file or replace this with your chosen license. Example: MIT.

## Built with

Python, Waitress, Flask, and Ollama.

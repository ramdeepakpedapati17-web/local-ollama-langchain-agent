# Local Ollama + LangChain Agent

This project runs a local chat agent using:
- **Ollama** as the local model server
- **LangChain** as the orchestration layer

## 1) Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com/download) installed

## 2) Start Ollama and pull a model

In a terminal:

```bash
ollama serve
```

Open another terminal and pull the model:

```bash
ollama pull llama3.1:8b
```

You can use another model too, but keep the model name in `.env` aligned.

## 3) Create and activate a virtual environment

From this project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 4) Install dependencies

```bash
pip install -r requirements.txt
```

## 5) Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Edit `.env` only if needed:
- `OLLAMA_BASE_URL`: URL where Ollama is running
- `OLLAMA_MODEL`: model to use

## 6) Run the agent

```bash
python agent.py
```

Then ask questions in the terminal. Type `exit` to quit.

## 7) Run the web app

Install dependencies first (includes Flask):

```bash
pip install -r requirements.txt
```

Start the web app:

```bash
python web_app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Session behavior in web UI:
- Leave session ID empty to start a new session automatically.
- Enter an existing session ID to continue that conversation.
- Session history is saved in `sessions/<session_id>.json`.

---

## What each file does

- `agent.py`  
  Builds a LangChain chain with:
  - a `system` message (agent behavior),
  - a `human` message template (`{input}`),
  - `ChatOllama` LLM as the backend.
- `requirements.txt`  
  Python dependencies.
- `.env.example`  
  Config template for model/server settings.

## How the flow works step by step

1. `load_dotenv()` loads your `.env` values.
2. `ChatOllama(...)` creates a client for your local Ollama model.
3. `ChatPromptTemplate.from_messages(...)` defines the prompt structure.
4. `prompt | llm` composes a runnable chain.
5. `chain.invoke({"input": user_input})` sends user text to the model.
6. The model response is printed to the console.

## Troubleshooting

- **Error: connection refused to localhost:11434**  
  Start Ollama with `ollama serve`.
- **Error: model not found**  
  Pull model first, e.g. `ollama pull llama3.1:8b`.
- **Very slow responses**  
  Try a smaller model (for example `llama3.2:3b`) and update `.env`.

import os
import json
import uuid
from pathlib import Path
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory


def build_agent():
    load_dotenv()

    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    llm = ChatOllama(
        model=model_name,
        base_url=base_url,
        temperature=0.2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a practical local AI assistant. "
                "Be concise, accurate, and explain your reasoning in short steps.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # prompt | llm creates a runnable chain in LangChain Expression Language.
    chain = prompt | llm
    store = {}
    sessions_dir = Path("sessions")
    sessions_dir.mkdir(exist_ok=True)

    def session_file_path(session_id: str) -> Path:
        return sessions_dir / f"{session_id}.json"

    def load_history_from_disk(session_id: str) -> InMemoryChatMessageHistory:
        history = InMemoryChatMessageHistory()
        file_path = session_file_path(session_id)
        if not file_path.exists():
            return history

        try:
            with file_path.open("r", encoding="utf-8") as f:
                records = json.load(f)
        except (json.JSONDecodeError, OSError):
            return history

        for record in records:
            role = record.get("type")
            content = record.get("content", "")
            if role == "human":
                history.add_message(HumanMessage(content=content))
            elif role == "ai":
                history.add_message(AIMessage(content=content))

        return history

    def save_history_to_disk(session_id: str):
        history = store.get(session_id)
        if history is None:
            return

        records = []
        for msg in history.messages:
            records.append({"type": msg.type, "content": msg.content})

        file_path = session_file_path(session_id)
        with file_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=True, indent=2)

    def get_history(session_id: str):
        if session_id not in store:
            store[session_id] = load_history_from_disk(session_id)
        return store[session_id]

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_history,
        input_messages_key="input",
        history_messages_key="history",
    )
    return chain_with_history, save_history_to_disk


def run_chat():
    chain, save_history_to_disk = build_agent()
    print("Local agent ready. Type your question, or type 'exit' to stop.")
    provided_session_id = input(
        "Enter a session ID to resume, or press Enter to start a new one: "
    ).strip()
    session_id = provided_session_id or str(uuid.uuid4())
    print(f"Using session_id: {session_id}")
    print(f"Session history file: sessions/{session_id}.json\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_input:
            continue

        response = chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        save_history_to_disk(session_id)
        print(f"\nAgent: {response.content}\n")


def chat_turn(chain, save_history_to_disk, session_id: str, user_input: str) -> str:
    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}},
    )
    save_history_to_disk(session_id)
    return response.content


if __name__ == "__main__":
    run_chat()

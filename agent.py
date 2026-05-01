import os
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


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
            ("human", "{input}"),
        ]
    )

    # prompt | llm creates a runnable chain in LangChain Expression Language.
    chain = prompt | llm
    return chain


def run_chat():
    chain = build_agent()
    print("Local agent ready. Type your question, or type 'exit' to stop.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Bye.")
            break
        if not user_input:
            continue

        response = chain.invoke({"input": user_input})
        print(f"\nAgent: {response.content}\n")


if __name__ == "__main__":
    run_chat()

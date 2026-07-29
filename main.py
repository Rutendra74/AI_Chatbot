from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a helpful AI assistant.

Use only the following context.

Context:
{context}

Question:
{question}

Answer:
"""

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model


def handle_conversation():

    history = []

    print("Welcome to the chatbot! Type 'exit' to quit.")

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # Keep only recent conversation
        context = "\n".join(history[-5:])

        result = chain.invoke(
            {
                "context": context,
                "question": user_input
            }
        )

        print("Bot:", result)

        history.append(f"User: {user_input}")
        history.append(f"AI: {result}")

        # Optional safety limit
        if len(history) > 20:
            history = history[-20:]


if __name__ == "__main__":
    handle_conversation()
from flask import Flask, render_template, request, session
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

app = Flask(__name__)
app.secret_key = 'Rutendra'  

template = '''
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
'''

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize chat history if not already in session
    if "chat_history" not in session:
        session["chat_history"] = ""

    bot_response = ""
    if request.method == "POST":
        user_input = request.form["user_input"]
        
        # Get the current chat history from the session
        context = session["chat_history"]
        
        # Get a response from the bot using the current chat history
        result = chain.invoke({"context": context, "question": user_input})
        
        # Append user input and bot response to the chat history
        session["chat_history"] += f"\nUser: {user_input}\nAI: {result}"
        
        # Set the bot response to show on the web page
        bot_response = result
    
    # Pass the full chat history to the template for display
    return render_template("index.html", response=bot_response, chat_history=session["chat_history"])

if __name__ == "__main__":
    app.run(debug=True)

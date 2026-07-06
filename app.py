from flask import Flask, render_template, request, session, redirect
from langchain_ollama import OllamaLLM
from werkzeug.security import check_password_hash,generate_password_hash
from langchain_core.prompts import ChatPromptTemplate
import sqlite3
app = Flask(__name__)
app.secret_key = 'Rutendra'  

template = '''
Answer the question below.

Here is the conversation history: {context}

Question: {question}

Answer:
'''

model = OllamaLLM(model="llama3.2")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        username=request.form['username']
        if not email or not password:
            return render_template('signup.html',error='All Fields are Required!!')
        hash_password=generate_password_hash(password)
        conn=sqlite3.connect('user.db')
        cursor=conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email=?',(email,))
        user=cursor.fetchone()
        if user is not None:
            return render_template('signup.html',error='User Already Exist')
            conn.close()
        cursor.execute('SELECT id from users WHERE username=?',(username,))
        user_id=cursor.fetchone()
        if user_id is not None:
            return render_template('signup.html',error='Username Already Exist!')
            conn.close()

        cursor.execute('INSERT INTO users(username,email,password_hash) VALUES (?,?,?)',(username,email,hash_password))
        conn.commit()
        conn.close()
        return render_template('login.html')
    
    return render_template('signup.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        conn=sqlite3.connect('user.db')
        cursor=conn.cursor()

        cursor.execute('SELECT id,password_hash FROM users WHERE email=?',(email,))
        user=cursor.fetchone()
        conn.close()

        if user is None:
            return render_template('login.html',error='Account not found! Please signup')
        user_id,password_hash=user
        if not check_password_hash(password_hash,password):
            return render_template('login.html',error='Invalid password')
        session['user_id']=user_id
        return redirect('/chat')
    return render_template('login.html')


@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route("/chat", methods=["GET", "POST"])
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

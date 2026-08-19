from flask import Flask, render_template, request, redirect,url_for,make_response,flash,Response
from langchain_ollama import OllamaLLM
from werkzeug.security import check_password_hash,generate_password_hash
from langchain_core.prompts import ChatPromptTemplate
import sqlite3
from werkzeug.utils import secure_filename
from flask_jwt_extended import (
    JWTManager,
    create_access_token,
    jwt_required,
    get_jwt_identity,
    unset_jwt_cookies,set_access_cookies)
from dotenv import load_dotenv
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
load_dotenv()
app = Flask(__name__)
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["100 per day", "30 per hour"]
)
app.secret_key=os.getenv('APP_SECRET_KEY')
app.config['JWT_SECRET_KEY']=os.getenv('JWT_SECRET_KEY')
app.config["JWT_TOKEN_LOCATION"] = ["cookies"]
app.config["JWT_COOKIE_CSRF_PROTECT"] = False
jwt=JWTManager(app)
TOP_K = int(os.getenv("TOP_K", 5))

template = """
You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not found in the context,
reply:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
model = OllamaLLM(model="llama3.2",base_url=os.getenv("OLLAMA_BASE_URL"))
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model
embed_model=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
@app.route('/signup',methods=['GET','POST'])
def signup():
    if request.method=='POST':
        email=request.form['email']
        password=request.form['password']
        username=request.form['username']
        if not email or not password:
            flash('All Fields are Required!!')
            return render_template('signup.html',error='All Fields are Required')
        hash_password=generate_password_hash(password)
        conn=sqlite3.connect('user.db')
        cursor=conn.cursor()
        cursor.execute('SELECT id FROM users WHERE email=?',(email,))
        user=cursor.fetchone()
        if user is not None:
            conn.close()
            flash('User Already Exist')
            return render_template('signup.html',error='Username Already Exist!')
        cursor.execute('SELECT id from users WHERE username=?',(username,))    
        cursor.execute('INSERT INTO users(username,email,password_hash) VALUES (?,?,?)',(username,email,hash_password))
        conn.commit()
        conn.close()
        return redirect('/login')
    
    return render_template('signup.html')

@app.route('/login',methods=['GET','POST'])
@limiter.limit("5 per minute")
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
            flash('Account not Found! please SignUp')
            return render_template('login.html',error='Account not found! Please signup')
        user_id,password_hash=user
        if not check_password_hash(password_hash,password):
            flash('Invalid Password')
            return render_template('login.html',error='Invalid password')
        access_token = create_access_token(identity=str(user_id))
        response=redirect('/chat')
        set_access_cookies(response,access_token)
        return response

    return render_template('login.html')
@app.route('/logout')
def logout():
    response=redirect(url_for('login'))
    unset_jwt_cookies(response)
    return response
def process_pdf(document_id,path):
    pdf=PdfReader(path)
    content=''
    for page in pdf.pages:
        temp=page.extract_text()
        if temp:
            content+=temp+"\n\n"
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=200)
    chunks=text_splitter.create_documents([content])
    texts=[chunk.page_content for chunk in chunks]
    embedding=embed_model.encode(texts)
    dimension=embedding.shape[1]
    index=faiss.IndexFlatL2(dimension)
    index.add(embedding)
    os.makedirs("vector_store", exist_ok=True)
    
    faiss.write_index(
    index,
    f"vector_store/{document_id}.index"
    )
    with open(f"vector_store/{document_id}_chunks.pkl", "wb") as file:
        pickle.dump(texts, file)
loaded_index,loaded_chunks={},{}
#cache for FAISS index and document chunks to remove the task everytime
def answer_question_stream(document_id, question):
    if document_id not in loaded_index:
        loaded_index[document_id] = faiss.read_index(
        f"vector_store/{document_id}.index"
    )
    if document_id not in loaded_chunks:
        with open(f"vector_store/{document_id}_chunks.pkl", "rb") as f:
            loaded_chunks[document_id] = pickle.load(f)

    index = loaded_index[document_id]
    texts = loaded_chunks[document_id]
    query_embed=embed_model.encode([question])
    distance,indices=index.search(query_embed,k=TOP_K)    
    retrieve_chunk=[texts[i] for i in indices[0] if i!=-1]
    context='\n\n'.join(retrieve_chunk)
    for chunk in chain.stream(
    {
        "context": context,
        "question": question
    }
):
        yield chunk
    

def answer_question(document_id,question):
    if document_id not in loaded_index:
        loaded_index[document_id] = faiss.read_index(
        f"vector_store/{document_id}.index"
    )
    if document_id not in loaded_chunks:
        with open(f"vector_store/{document_id}_chunks.pkl", "rb") as f:
            loaded_chunks[document_id] = pickle.load(f)

    index = loaded_index[document_id]
    texts = loaded_chunks[document_id]
    query_embed=embed_model.encode([question])
    distance,indices=index.search(query_embed,k=TOP_K)    
    retrieve_chunk=[texts[i] for i in indices[0] if i!=-1]
    context='\n\n'.join(retrieve_chunk)
    response = chain.invoke(
    {
        "context": context,
        "question": question
    }
)
    return response



@app.route('/upload',methods=['POST'])
@jwt_required()
@limiter.limit("10 per hour")
def upload():
    file=request.files.get('pdf_file')
    user_id=get_jwt_identity()
    if not file:
        return render_template('index.html',error='File not Uploaded!!')
    file_name=secure_filename(file.filename)
    if not file_name.lower().endswith('.pdf'):
        return render_template('index.html',error='Please Insert pdf files only')
    path=os.path.join('uploads',file_name)
    os.makedirs('uploads',exist_ok=True)
    file.save(path)
    conn=sqlite3.connect('user.db')
    cursor=conn.cursor()
    cursor.execute('INSERT INTO document(document_name,file_path,user_id) VALUES (?,?,?)',(file_name,path,user_id))
    document_id=cursor.lastrowid
    
    conn.commit()
    conn.close()
    process_pdf(document_id,path)
    return redirect(url_for('index'))
@app.route('/',methods=['GET','POST'])
def home():
    return render_template('home.html')

@app.route("/chat", methods=["GET", "POST"])
@jwt_required()
def index():
    user_id = get_jwt_identity()
    document_id=None
    filename=None
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()
    cursor.execute(
            """
            SELECT id,document_name
            FROM document
            WHERE user_id=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (user_id,)
        )
    document = cursor.fetchone()
    if document:
        document_id=document[0]
        filename=document[1]

    bot_result = ""
    if request.method == "POST":
        user_message = request.form["user_input"]

        # Save user's message
        cursor.execute(
            "INSERT INTO messages(user_id, role, content) VALUES (?,?,?)",
            (user_id, "User", user_message)
        )
        
        
        if document_id is not None:
            result = answer_question(document_id, user_message)
        else:
            result = "Please upload a PDF before asking questions."
        # Save AI response
        cursor.execute(
            "INSERT INTO messages(user_id, role, content) VALUES (?,?,?)",
            (user_id, "AI", result)
        )
        conn.commit()
        bot_result = result
    # Load chat history
    cursor.execute(
        """
        SELECT role, content
        FROM messages
        WHERE user_id=?
        ORDER BY created_at
        """,
        (user_id,)
    )
    
    rows = cursor.fetchall()
    
    cursor.execute('SELECT username FROM users WHERE id=?',(user_id,))
    username=cursor.fetchone()[0]
    conn.close()

    response = make_response(
    render_template(
        "index.html",
        response=bot_result,
        rows=rows,
        filename=filename,
        username=username
    )
)

    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"    
    return response
@app.route("/chat/stream", methods=["POST"])
@jwt_required()
def chat_stream():
    user_id = get_jwt_identity()
    user_message = request.form["user_input"]

    # First connection: only to fetch document_id
    conn = sqlite3.connect("user.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT id
        FROM document
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (user_id,)
    )

    document = cursor.fetchone()
    conn.close()          # <-- close it immediately

    if not document:
        return Response(
            "Please upload a PDF before asking questions.",
            mimetype="text/plain"
        )

    document_id = document[0]

    def generate():

        # Second connection: only for saving messages
        conn = sqlite3.connect("user.db")
        cursor = conn.cursor()

        full_response = ""

        for chunk in answer_question_stream(document_id, user_message):
            full_response += chunk
            yield chunk

        cursor.execute(
            "INSERT INTO messages(user_id, role, content) VALUES (?,?,?)",
            (user_id, "User", user_message)
        )

        cursor.execute(
            "INSERT INTO messages(user_id, role, content) VALUES (?,?,?)",
            (user_id, "AI", full_response)
        )

        conn.commit()
        conn.close()

    return Response(
        generate(),
        mimetype="text/plain")
@jwt.unauthorized_loader
def unauthorized_callback(reason):
    return redirect(url_for("login"))

@jwt.invalid_token_loader
def invalid_token_callback(reason):
    return redirect(url_for("login"))

@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return redirect(url_for("login"))
if __name__ == "__main__":
    
    app.run(host="0.0.0.0", port=5000,debug=True)

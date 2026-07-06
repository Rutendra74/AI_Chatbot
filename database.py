import sqlite3
conn=sqlite3.connect('user.db')
cursor=conn.cursor()

cursor.execute('''CREATE TABLE IF NOT EXISTS users(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL
)''')
cursor.execute('''CREATE TABLE IF NOT EXISTS messages(id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) )''')

cursor.execute(
    """
    INSERT INTO users(username,email,password_hash)
    VALUES(?,?,?)
    """,
    ("Rutendra","test@gmail.com","dummyhash")
)
conn.commit()
conn.close()
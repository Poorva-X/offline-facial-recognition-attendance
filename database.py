import sqlite3
from datetime import datetime

def create_database():
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    # Create employees table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            emp_id TEXT UNIQUE NOT NULL,
            registered_date TEXT
        )
    ''')
    
    # Create attendance table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            emp_id TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            status TEXT DEFAULT 'Present'
        )
    ''')
    
    conn.commit()
    conn.close()
    print("✅ Database created successfully")

if __name__ == "__main__":
    create_database()
    print("Database is ready")
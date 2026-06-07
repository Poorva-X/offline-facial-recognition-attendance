from google import genai
from datetime import datetime
import sqlite3
import time

# Configure Gemini
import os
from dotenv import load_dotenv
load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_attendance_history(emp_id):
    conn = sqlite3.connect('attendance.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT date, time, status 
        FROM attendance 
        WHERE emp_id = ? 
        ORDER BY date DESC 
        LIMIT 7
    ''', (emp_id,))
    
    records = cursor.fetchall()
    conn.close()
    return records

def think_and_act(name, emp_id, current_time):
    time.sleep(2)
    history = get_attendance_history(emp_id)
    
    history_text = ""
    for record in history:
        history_text += f"Date: {record[0]} | Time: {record[1]} | Status: {record[2]}\n"
    
    if not history_text:
        history_text = "No previous attendance records found."
    
    hour = int(current_time.split(":")[0])
    is_late = hour >= 9
    
    prompt = f"""
    You are an intelligent attendance management agent.
    
    Current situation:
    - Employee Name: {name}
    - Employee ID: {emp_id}
    - Current Time: {current_time}
    - Is Late (after 9 AM): {is_late}
    
    Last 7 days attendance history:
    {history_text}
    
    Based on this information analyze the attendance 
    pattern and respond in this exact format:
    STATUS: (Normal/Late/Frequent Late/Suspicious)
    ACTION: (what should be done)
    MESSAGE: (brief message for supervisor)
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

def handle_unknown_person():
    time.sleep(2)
    prompt = """
    You are a security agent for an attendance system.
    
    An unknown person just tried to mark attendance but 
    their face was not recognized in the database.
    
    Generate a security alert message for the supervisor.
    Keep it brief and professional.
    
    Format:
    ALERT: Security Alert
    ACTION: (what supervisor should do)
    MESSAGE: (brief alert message)
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    return response.text

if __name__ == "__main__":
    print("Testing agent with known person...")
    result = think_and_act("Rahul Sharma", "EMP001", "10:30:00")
    print(result)
    print("\nTesting agent with unknown person...")
    result = handle_unknown_person()
    print(result)
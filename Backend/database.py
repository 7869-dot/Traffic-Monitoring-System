import sqlite3
from contextlib import contextmanager
from typing import Generator
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), "Traffic.db")

@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager for database connections.
    Ensures proper connection handling and cleanup.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Enable column access by name
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

def init_db():
    """
    Initialize the database with required tables.
    """
    with get_db() as conn:
        cursor = conn.cursor()
        
        # Traffic data table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS traffic_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_count INTEGER NOT NULL,
                location TEXT,
                camera_id TEXT,
                traffic_status TEXT
            )
        """)
        
        # Arduino control log table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS arduino_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                status TEXT,
                details TEXT
            )
        """)
        
        # Vehicle detection records
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vehicle_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_type TEXT,
                confidence REAL,
                frame_data BLOB,
                camera_id TEXT
            )
        """)
        
        conn.commit()
        print("Database tables initialized successfully")


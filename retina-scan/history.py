"""
Local patient history tracking for longitudinal analysis.
Stores scan results in a local SQLite database and generates trend graphs.
"""

import sqlite3
import datetime
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

DB_PATH = Path("patient_history.db")

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.execute('''
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id TEXT,
            timestamp DATETIME,
            dr_grade INTEGER,
            vcdr REAL,
            avr REAL,
            media_clarity INTEGER
        )
    ''')
    return conn

def save_scan(patient_id: str, dr_grade: int, vcdr: float, avr: float, media_clarity: int):
    if not patient_id:
        return
    
    conn = _get_conn()
    with conn:
        conn.execute('''
            INSERT INTO scans (patient_id, timestamp, dr_grade, vcdr, avr, media_clarity)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (patient_id, datetime.datetime.now(), dr_grade, vcdr, avr, media_clarity))
    conn.close()

def get_history_df(patient_id: str) -> pd.DataFrame:
    conn = _get_conn()
    df = pd.read_sql_query(
        "SELECT * FROM scans WHERE patient_id = ? ORDER BY timestamp ASC",
        conn, params=(patient_id,)
    )
    conn.close()
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def plot_progression(df: pd.DataFrame):
    """
    Returns a Plotly figure showing disease progression over time.
    """
    if df.empty:
        return None

    # Create figure with secondary y-axis
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['dr_grade'], name="DR Grade", 
                   mode='lines+markers', line=dict(color='#d73027', width=3)),
        secondary_y=False,
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['vcdr'], name="Glaucoma Risk (VCDR)", 
                   mode='lines+markers', line=dict(color='#1a9850', dash='dash')),
        secondary_y=True,
    )
    
    fig.add_trace(
        go.Scatter(x=df['timestamp'], y=df['avr'], name="Hypertensive Risk (AVR)", 
                   mode='lines+markers', line=dict(color='#4575b4', dash='dot')),
        secondary_y=True,
    )

    fig.update_layout(
        title="Disease Progression Over Time",
        xaxis_title="Scan Date",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    fig.update_yaxes(title_text="DR Severity Grade (0-4)", range=[0, 4.5], dtick=1, secondary_y=False)
    fig.update_yaxes(title_text="Ratio (VCDR / AVR)", range=[0, 1.2], secondary_y=True)

    return fig

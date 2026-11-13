import sqlite3
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from backend.core.config import settings

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.database_path), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # Datasets Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        source TEXT,
        license TEXT,
        languages TEXT,
        modalities TEXT,
        file_path TEXT,
        sample_count INTEGER DEFAULT 0,
        class_distribution TEXT,
        validation_report TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Model Registry Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_registry (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        architecture TEXT NOT NULL,
        version TEXT NOT NULL,
        checkpoint_path TEXT,
        status TEXT DEFAULT 'ready',
        is_active INTEGER DEFAULT 0,
        metrics TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Model Versions Table (Audit & Rollback)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_versions (
        version_id TEXT PRIMARY KEY,
        model_id TEXT NOT NULL,
        parent_version TEXT,
        optimization_method TEXT,
        reward_score REAL,
        macro_f1 REAL,
        accuracy REAL,
        checkpoint_path TEXT,
        eval_details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Predictions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id TEXT PRIMARY KEY,
        input_text TEXT,
        image_path TEXT,
        context_data TEXT,
        predicted_label INTEGER NOT NULL,
        confidence REAL NOT NULL,
        model_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        language_detected TEXT,
        modalities_used TEXT,
        explanation_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Feedback Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id TEXT PRIMARY KEY,
        prediction_id TEXT NOT NULL,
        is_correct INTEGER,
        suggested_label INTEGER,
        explanation_rating INTEGER, -- 1 to 5 scale
        flags TEXT, -- json list of tags e.g. ["context_misunderstood", "sarcasm"]
        comment TEXT,
        status TEXT DEFAULT 'pending', -- pending, accepted, rejected
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (prediction_id) REFERENCES predictions(id)
    )
    """)

    # Rewards Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS rewards (
        id TEXT PRIMARY KEY,
        prediction_id TEXT NOT NULL,
        feedback_id TEXT,
        r_classification REAL,
        r_explanation REAL,
        r_context REAL,
        r_feedback REAL,
        r_error REAL,
        total_reward REAL,
        details TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Optimization Runs Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS optimization_runs (
        id TEXT PRIMARY KEY,
        run_name TEXT NOT NULL,
        base_version TEXT NOT NULL,
        new_version TEXT NOT NULL,
        method TEXT NOT NULL, -- e.g. "DPO", "Preference-SFT"
        feedback_samples_used INTEGER,
        status TEXT DEFAULT 'completed',
        loss_history TEXT,
        metrics_before TEXT,
        metrics_after TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Experiments Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS experiments (
        id TEXT PRIMARY KEY,
        exp_code TEXT NOT NULL,
        title TEXT NOT NULL,
        status TEXT DEFAULT 'pending',
        config_json TEXT,
        results_json TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Background Tasks Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS background_tasks (
        task_id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        status TEXT DEFAULT 'running', -- running, completed, failed
        progress REAL DEFAULT 0.0,
        result_json TEXT,
        error_message TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database schema successfully initialized.")


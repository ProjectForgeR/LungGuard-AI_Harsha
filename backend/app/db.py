"""
Database access layer for LungGuard AI.
Uses persistent SQLite storage to record all CT scan and breathing test results.
"""

import os
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

DB_DIR = Path(__file__).resolve().parent.parent.parent / "data"
DB_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = DB_DIR / "lungguard.db"


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """Initialize database tables."""
    with get_db_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS test_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                test_id TEXT UNIQUE NOT NULL,
                user_id TEXT DEFAULT 'patient_guest',
                test_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                test_date TEXT NOT NULL,
                test_time TEXT NOT NULL,
                detection_result TEXT NOT NULL,
                cancer_status TEXT DEFAULT 'N/A',
                cancer_stage TEXT DEFAULT 'N/A',
                confidence REAL DEFAULT 0.0,
                image_url TEXT DEFAULT '',
                abnormality_description TEXT DEFAULT '',
                scan_summary TEXT DEFAULT '',
                recommended_next_steps TEXT DEFAULT '',
                raw_metadata TEXT DEFAULT '{}'
            );
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_test_id ON test_records(test_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON test_records(created_at);")
        conn.commit()


def generate_test_id(prefix: str = "CT") -> str:
    """Generate a unique human-readable test ID, e.g. LUNG-CT-20261108-A1B2."""
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:4].upper()
    return f"LUNG-{prefix}-{date_str}-{short_uuid}"


def save_test_record(
    test_type: str,
    detection_result: str,
    cancer_status: str = "N/A",
    cancer_stage: str = "N/A",
    confidence: float = 0.0,
    image_url: str = "",
    abnormality_description: str = "",
    scan_summary: str = "",
    recommended_next_steps: str = "",
    raw_metadata: Optional[Dict[str, Any]] = None,
    user_id: str = "patient_guest"
) -> str:
    """
    Saves an actual completed test record to the persistent database.
    Returns the unique test_id.
    """
    init_db()
    prefix = "CT" if test_type == "CT Scan" else "BR"
    test_id = generate_test_id(prefix)
    now = datetime.now()
    created_at = now.isoformat()
    test_date = now.strftime("%Y-%m-%d")
    test_time = now.strftime("%H:%M:%S")

    metadata_str = json.dumps(raw_metadata or {})
    if isinstance(recommended_next_steps, list):
        steps_str = "\n".join(str(s) for s in recommended_next_steps)
    else:
        steps_str = str(recommended_next_steps or "")

    with get_db_connection() as conn:
        conn.execute("""
            INSERT INTO test_records (
                test_id, user_id, test_type, created_at, test_date, test_time,
                detection_result, cancer_status, cancer_stage, confidence,
                image_url, abnormality_description, scan_summary,
                recommended_next_steps, raw_metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            test_id, user_id, test_type, created_at, test_date, test_time,
            detection_result, cancer_status, cancer_stage, float(confidence),
            image_url, abnormality_description, scan_summary,
            steps_str, metadata_str
        ))
        conn.commit()

    return test_id


def get_dashboard_stats(user_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Calculates actual statistical counts from stored database records.
    """
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # Total CT Scans
        cursor.execute("SELECT COUNT(*) FROM test_records WHERE test_type = 'CT Scan'")
        total_ct = cursor.fetchone()[0]

        # Cancer Detected
        cursor.execute("""
            SELECT COUNT(*) FROM test_records 
            WHERE test_type = 'CT Scan' 
              AND (detection_result LIKE '%Cancer Detected%' AND detection_result NOT LIKE '%No%')
        """)
        cancer_detected = cursor.fetchone()[0]

        # No Cancer Detected
        cursor.execute("""
            SELECT COUNT(*) FROM test_records 
            WHERE test_type = 'CT Scan' 
              AND (detection_result LIKE '%No Lung Cancer%' OR detection_result LIKE '%No Cancer%')
        """)
        no_cancer_detected = cursor.fetchone()[0]

        # Inconclusive Results
        cursor.execute("""
            SELECT COUNT(*) FROM test_records 
            WHERE test_type = 'CT Scan' 
              AND detection_result LIKE '%Inconclusive%'
        """)
        inconclusive = cursor.fetchone()[0]

        # Total Breathing Assessments
        cursor.execute("SELECT COUNT(*) FROM test_records WHERE test_type = 'Breathing Test'")
        total_breathing = cursor.fetchone()[0]

        # Total All Tests
        cursor.execute("SELECT COUNT(*) FROM test_records")
        total_tests = cursor.fetchone()[0]

        # Recent breakdown (last 7 days activity)
        cursor.execute("""
            SELECT test_date, COUNT(*) as count 
            FROM test_records 
            GROUP BY test_date 
            ORDER BY test_date DESC 
            LIMIT 7
        """)
        daily_rows = cursor.fetchall()
        daily_activity = [{"date": r["test_date"], "count": r["count"]} for r in reversed(daily_rows)]

        return {
            "total_ct_scans": total_ct,
            "cancer_detected": cancer_detected,
            "no_cancer_detected": no_cancer_detected,
            "inconclusive": inconclusive,
            "total_breathing_assessments": total_breathing,
            "total_tests": total_tests,
            "daily_activity": daily_activity
        }


def list_test_records(
    test_type: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """List test records with optional filtering."""
    init_db()
    query = "SELECT * FROM test_records WHERE 1=1"
    params: List[Any] = []

    if test_type and test_type != "all":
        query += " AND test_type = ?"
        params.append(test_type)

    if search:
        query += " AND (test_id LIKE ? OR detection_result LIKE ? OR test_date LIKE ? OR cancer_status LIKE ?)"
        s = f"%{search}%"
        params.extend([s, s, s, s])

    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()

        results = []
        for r in rows:
            record = dict(r)
            try:
                record["raw_metadata"] = json.loads(record.get("raw_metadata") or "{}")
            except Exception:
                record["raw_metadata"] = {}
            results.append(record)

        return results


def get_test_record_by_id(test_id: str) -> Optional[Dict[str, Any]]:
    """Retrieve full details of an individual test record."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM test_records WHERE test_id = ?", (test_id,))
        row = cursor.fetchone()
        if not row:
            return None
        record = dict(row)
        try:
            record["raw_metadata"] = json.loads(record.get("raw_metadata") or "{}")
        except Exception:
            record["raw_metadata"] = {}
        return record


def delete_test_record(test_id: str) -> bool:
    """Deletes a test record from the database."""
    init_db()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM test_records WHERE test_id = ?", (test_id,))
        conn.commit()
        return cursor.rowcount > 0

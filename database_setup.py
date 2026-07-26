import sqlite3


DB_FILE = "users.db"


def add_column_if_missing(cursor, table, column, column_type):
    cursor.execute(f"PRAGMA table_info({table})")
    existing_columns = {row[1] for row in cursor.fetchall()}
    if column not in existing_columns:
        cursor.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")


def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout = 10000")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            staff_id TEXT,
            password TEXT NOT NULL,
            role TEXT DEFAULT 'administrator',
            first_login INTEGER DEFAULT 1,
            password_hashed INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS student_records (
            record_id INTEGER PRIMARY KEY AUTOINCREMENT,
            uploaded_by TEXT,
            student_name TEXT,
            student_id TEXT,
            age REAL,
            study_hours REAL,
            attendance_rate REAL,
            assignment_delay REAL,
            semester TEXT,
            gpa REAL,
            internet_access TEXT,
            part_time_job TEXT,
            family_problems TEXT,
            family_reason TEXT,
            scholarship TEXT,
            department TEXT,
            stress_index REAL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS prediction_history (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT,
            student_name TEXT,
            predicted_risk TEXT,
            probability_score REAL,
            semester TEXT,
            gpa REAL,
            stress_index REAL,
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            predicted_by TEXT,
            input_method TEXT,
            top_factors TEXT,
            interventions TEXT,
            ai_suggestions TEXT
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS authorized_users (
            authorized_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            staff_id TEXT UNIQUE NOT NULL,
            role TEXT NOT NULL,
            is_active INTEGER DEFAULT 1,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            user_role TEXT,
            action_type TEXT,
            action_status TEXT,
            action_details TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    migrations = {
        "users": {
            "staff_id": "TEXT",
            "role": "TEXT DEFAULT 'administrator'",
            "first_login": "INTEGER DEFAULT 1",
            "password_hashed": "INTEGER DEFAULT 1",
        },
        "student_records": {
            "family_reason": "TEXT",
        },
        "prediction_history": {
            "input_method": "TEXT",
            "top_factors": "TEXT",
            "interventions": "TEXT",
            "ai_suggestions": "TEXT",
            "predicted_by": "TEXT",
        },
        "authorized_users": {
            "is_active": "INTEGER DEFAULT 1",
        },
    }

    for table, columns in migrations.items():
        for column, column_type in columns.items():
            add_column_if_missing(cursor, table, column, column_type)

    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_staff_id
        ON users(staff_id)
        WHERE staff_id IS NOT NULL AND staff_id != ''
        """
    )
    cursor.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_authorized_users_staff_id
        ON authorized_users(staff_id)
        WHERE staff_id IS NOT NULL AND staff_id != ''
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_student_id "
        "ON prediction_history(student_id)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_prediction_history_time "
        "ON prediction_history(prediction_time)"
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_database()
    print("AURA database setup completed successfully.")

from datetime import datetime, timedelta, timezone

import pandas as pd


SEMESTER_ORDER = {"February": 1, "April": 2, "September": 3}
VALID_UPLOAD_TYPES = ["Final", "Resit"]
PASS_MARK = 50.0
APP_TIMEZONE = timezone(timedelta(hours=8))


def app_now():
    return datetime.now(APP_TIMEZONE).replace(tzinfo=None)


def app_now_text():
    return app_now().strftime("%Y-%m-%d %H:%M:%S")


def normalize_student_id(value):
    text = str(value or "").strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text.upper()


def normalize_semester_term(value):
    text = str(value or "").strip().title()
    aliases = {
        "Feb": "February",
        "February": "February",
        "Apr": "April",
        "April": "April",
        "Sep": "September",
        "Sept": "September",
        "September": "September",
    }
    return aliases.get(text, text)


def normalize_upload_type(value):
    text = str(value or "").strip().title()
    return text if text in VALID_UPLOAD_TYPES else "Final"


def result_status(mark):
    try:
        return "Fail" if float(mark) < PASS_MARK else "Pass"
    except (TypeError, ValueError):
        return "Invalid"


def setup_course_tables(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS course_catalog (
            course_code TEXT PRIMARY KEY,
            course_name TEXT NOT NULL,
            level TEXT,
            programme TEXT,
            category TEXT,
            year TEXT,
            specialisation TEXT,
            source_status TEXT,
            source_url TEXT,
            imported_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS course_results_raw (
            result_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT,
            course_code TEXT NOT NULL,
            course_name TEXT NOT NULL,
            academic_year INTEGER NOT NULL,
            semester_term TEXT NOT NULL,
            upload_type TEXT NOT NULL,
            course_mark REAL NOT NULL,
            result_status TEXT NOT NULL,
            uploaded_by TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS course_results_current (
            current_id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT,
            course_code TEXT NOT NULL,
            course_name TEXT NOT NULL,
            academic_year INTEGER NOT NULL,
            semester_term TEXT NOT NULL,
            final_mark REAL NOT NULL,
            final_status TEXT NOT NULL,
            last_upload_type TEXT NOT NULL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(student_id, course_code, academic_year, semester_term)
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_course_results_current_semester "
        "ON course_results_current(academic_year, semester_term, final_status)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_course_results_current_course "
        "ON course_results_current(course_code, course_name)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_course_results_raw_semester "
        "ON course_results_raw(academic_year, semester_term, upload_type)"
    )


def sync_course_catalog_from_csv(conn, catalog_file):
    if not catalog_file.exists():
        return 0

    df = pd.read_csv(catalog_file, dtype=str, keep_default_na=False).fillna("")
    required = {"course_code", "course_name"}
    if not required.issubset(df.columns):
        return 0

    optional_columns = [
        "level",
        "programme",
        "category",
        "year",
        "specialisation",
        "source_status",
        "source_url",
    ]
    rows = []
    for _, row in df.iterrows():
        course_code = str(row.get("course_code", "")).strip().upper()
        course_name = str(row.get("course_name", "")).strip()
        if not course_code or not course_name:
            continue
        rows.append(
            (
                course_code,
                course_name,
                *[str(row.get(column, "")).strip() for column in optional_columns],
            )
        )

    conn.executemany(
        """
        INSERT OR REPLACE INTO course_catalog (
            course_code, course_name, level, programme, category, year,
            specialisation, source_status, source_url
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def normalize_course_results_dataframe(df, academic_year, semester_term, upload_type):
    column_aliases = {
        "student id": "student_id",
        "studentid": "student_id",
        "student name": "student_name",
        "course code": "course_code",
        "course name": "course_name",
        "year": "academic_year",
        "academic year": "academic_year",
        "semester": "semester_term",
        "semester term": "semester_term",
        "term": "semester_term",
        "upload type": "upload_type",
        "result type": "upload_type",
        "mark": "course_mark",
        "course mark": "course_mark",
        "course_mark": "course_mark",
    }
    normalized = df.copy()
    normalized.columns = [
        column_aliases.get(str(column).strip().lower(), str(column).strip())
        for column in normalized.columns
    ]

    if "academic_year" not in normalized.columns:
        normalized["academic_year"] = academic_year
    if "semester_term" not in normalized.columns:
        normalized["semester_term"] = semester_term
    if "upload_type" not in normalized.columns:
        normalized["upload_type"] = upload_type

    required = [
        "student_id",
        "student_name",
        "course_code",
        "course_name",
        "academic_year",
        "semester_term",
        "upload_type",
        "course_mark",
    ]
    missing = [column for column in required if column not in normalized.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    normalized = normalized[required].copy()
    normalized["student_id"] = normalized["student_id"].apply(normalize_student_id)
    normalized["student_name"] = normalized["student_name"].astype(str).str.strip()
    normalized["course_code"] = normalized["course_code"].astype(str).str.strip().str.upper()
    normalized["course_name"] = normalized["course_name"].astype(str).str.strip()
    normalized["academic_year"] = pd.to_numeric(
        normalized["academic_year"], errors="coerce"
    ).astype("Int64")
    normalized["semester_term"] = normalized["semester_term"].apply(normalize_semester_term)
    normalized["upload_type"] = normalized["upload_type"].apply(normalize_upload_type)
    normalized["course_mark"] = pd.to_numeric(normalized["course_mark"], errors="coerce")
    normalized = normalized.dropna(subset=["academic_year", "course_mark"])
    normalized = normalized[
        (normalized["student_id"] != "")
        & (normalized["course_code"] != "")
        & (normalized["course_name"] != "")
    ].copy()
    normalized["academic_year"] = normalized["academic_year"].astype(int)
    normalized["result_status"] = normalized["course_mark"].apply(result_status)
    return normalized


def save_course_results(conn, df, uploaded_by, academic_year, semester_term, upload_type):
    normalized = normalize_course_results_dataframe(
        df, academic_year, semester_term, upload_type
    )
    if normalized.empty:
        return 0, 0

    now = app_now_text()
    raw_rows = []
    current_rows = []
    for _, row in normalized.iterrows():
        raw_rows.append(
            (
                row["student_id"],
                row["student_name"],
                row["course_code"],
                row["course_name"],
                int(row["academic_year"]),
                row["semester_term"],
                row["upload_type"],
                float(row["course_mark"]),
                row["result_status"],
                uploaded_by,
                now,
            )
        )
        current_rows.append(
            (
                row["student_id"],
                row["student_name"],
                row["course_code"],
                row["course_name"],
                int(row["academic_year"]),
                row["semester_term"],
                float(row["course_mark"]),
                row["result_status"],
                row["upload_type"],
                now,
            )
        )

    conn.executemany(
        """
        INSERT INTO course_results_raw (
            student_id, student_name, course_code, course_name, academic_year,
            semester_term, upload_type, course_mark, result_status, uploaded_by,
            uploaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        raw_rows,
    )
    conn.executemany(
        """
        INSERT INTO course_results_current (
            student_id, student_name, course_code, course_name, academic_year,
            semester_term, final_mark, final_status, last_upload_type, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id, course_code, academic_year, semester_term)
        DO UPDATE SET
            student_name = excluded.student_name,
            course_name = excluded.course_name,
            final_mark = excluded.final_mark,
            final_status = excluded.final_status,
            last_upload_type = excluded.last_upload_type,
            updated_at = excluded.updated_at
        """,
        current_rows,
    )
    conn.commit()
    return len(raw_rows), len(current_rows)


def load_course_results_current(conn):
    return pd.read_sql_query(
        """
        SELECT *
        FROM course_results_current
        ORDER BY academic_year DESC, updated_at DESC
        """,
        conn,
    )


def semester_sort_key(row):
    return (
        int(row.get("academic_year", 0)),
        SEMESTER_ORDER.get(str(row.get("semester_term", "")), 0),
    )


def latest_year_semester(df):
    if df.empty:
        return None, None
    pairs = (
        df[["academic_year", "semester_term"]]
        .drop_duplicates()
        .to_dict("records")
    )
    latest = max(pairs, key=semester_sort_key)
    return int(latest["academic_year"]), str(latest["semester_term"])


def course_failure_summary(df):
    if df.empty:
        return pd.DataFrame(
            columns=[
                "course_code",
                "course_name",
                "total_students",
                "fail_number",
                "fail_rate",
            ]
        )
    grouped = (
        df.groupby(["course_code", "course_name"], as_index=False)
        .agg(
            total_students=("student_id", "nunique"),
            fail_number=("final_status", lambda values: int((values == "Fail").sum())),
        )
    )
    grouped["fail_rate"] = (
        grouped["fail_number"] / grouped["total_students"].replace(0, pd.NA) * 100
    ).fillna(0)
    return grouped.sort_values(
        ["fail_number", "fail_rate", "course_name"], ascending=[False, False, True]
    )


def course_trend_summary(df, course_search):
    if df.empty or not course_search:
        return pd.DataFrame()
    search = str(course_search).strip()
    mask = (
        df["course_name"].astype(str).str.contains(search, case=False, na=False, regex=False)
        | df["course_code"].astype(str).str.contains(search, case=False, na=False, regex=False)
    )
    filtered = df[mask].copy()
    if filtered.empty:
        return pd.DataFrame()
    grouped = (
        filtered.groupby(["academic_year", "semester_term"], as_index=False)
        .agg(
            total_students=("student_id", "nunique"),
            fail_number=("final_status", lambda values: int((values == "Fail").sum())),
        )
    )
    grouped["fail_rate"] = (
        grouped["fail_number"] / grouped["total_students"].replace(0, pd.NA) * 100
    ).fillna(0)
    grouped["_semester_order"] = grouped["semester_term"].map(SEMESTER_ORDER).fillna(0)
    return grouped.sort_values(["academic_year", "_semester_order"]).drop(
        columns=["_semester_order"]
    )

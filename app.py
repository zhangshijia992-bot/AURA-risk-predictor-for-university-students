import json
import html
import base64
import os
import random
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import bcrypt
import joblib
import pandas as pd
import plotly.express as px
import plotly.io as pio
import streamlit as st
from zhipuai import ZhipuAI

import notification_service as notifications
import course_failure_service as course_failures

st.set_page_config(
    page_title="AURA Student Risk Prediction",
    page_icon="aura_logo.jpeg",
    layout="wide",
)

pio.templates.default = "plotly_white"


def aura_plotly_chart(fig, **kwargs):
    """Render Plotly charts with the same readable light card style across the app."""
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font={
            "family": "Aptos, Segoe UI, Inter, sans-serif",
            "color": "#172033",
            "size": 13,
        },
        title_font={"color": "#172033"},
        legend={
            "font": {"color": "#172033"},
            "title": {"font": {"color": "#172033"}},
            "bgcolor": "rgba(255,255,255,0.88)",
            "bordercolor": "rgba(23,32,51,0.10)",
            "borderwidth": 1,
        },
        margin={"l": 54, "r": 28, "t": 48, "b": 58},
        hoverlabel={
            "bgcolor": "#ffffff",
            "bordercolor": "#cfd1d8",
            "font": {"color": "#172033"},
        },
    )
    fig.update_xaxes(
        color="#172033",
        title_font={"color": "#172033"},
        tickfont={"color": "#172033"},
        gridcolor="#e8eaf0",
        linecolor="#cfd1d8",
        zerolinecolor="#d8dbe4",
        automargin=True,
    )
    fig.update_yaxes(
        color="#172033",
        title_font={"color": "#172033"},
        tickfont={"color": "#172033"},
        gridcolor="#e8eaf0",
        linecolor="#cfd1d8",
        zerolinecolor="#d8dbe4",
        automargin=True,
    )
    if "use_container_width" not in kwargs:
        kwargs["use_container_width"] = True
    return st.plotly_chart(fig, theme=None, **kwargs)


THIS_FILE = Path(__file__).resolve()
PROJECT_DIR = THIS_FILE.parent.parent if THIS_FILE.parent.name == "fixed_versions" else THIS_FILE.parent
DB_FILE = PROJECT_DIR / "users.db"
MODEL_FILE = PROJECT_DIR / "best_randomforest_model_v2.pkl"
MODEL_TRAINING_DATA_FILE = PROJECT_DIR / "improved_dropout_dataset.csv"
LOGO_FILE = PROJECT_DIR / "aura_logo.jpeg"
AUTHORIZED_USERS_FILE = PROJECT_DIR / "authorized_users.csv"
STUDENT_CONTACTS_FILE = PROJECT_DIR / "student_contacts.csv"
COURSE_CATALOG_FILE = PROJECT_DIR / "course_catalog.csv"
APP_TIMEZONE = timezone(timedelta(hours=8))


def app_now():
    return datetime.now(APP_TIMEZONE).replace(tzinfo=None)


def app_now_text():
    return app_now().strftime("%Y-%m-%d %H:%M:%S")

MODEL_FEATURE_NAMES = [
    "Age",
    "Study_Hours_per_Day",
    "Attendance_Rate",
    "Assignment_Delay_Days",
    "Stress_Index",
    "Internet_Access_Yes",
    "Part_Time_Job_Yes",
    "Scholarship_Yes",
    "Semester_Year 2",
    "Semester_Year 3",
    "Semester_Year 4",
    "Department_Business",
    "Department_CS",
    "Department_Engineering",
    "Department_Science",
]

ADMIN_EMAIL = st.secrets.get("ADMIN_EMAIL", "admin@aura-student-risk.com").strip().lower()
ROLE_EMAIL_DOMAINS = {
    "administrator": "aura-student-risk.com",
    "admin": "aura-student-risk.com",
    "advisor": "aura-student-risk.com",
}
GEMINI_API_KEY = (
    st.secrets.get("GEMINI_API_KEY", "").strip()
    or os.environ.get("GOOGLE_API_KEY", "").strip()
    or os.environ.get("GEMINI_API_KEY", "").strip()
)
GEMINI_MODEL = st.secrets.get(
    "GEMINI_MODEL", "gemini-3.1-flash-lite"
).strip()
RESEND_API_KEY = st.secrets.get("RESEND_API_KEY", "").strip()
EMAIL_FROM = st.secrets.get("EMAIL_FROM", "onboarding@aura-student-risk.com").strip()
EMAIL_SENDER_NAME = st.secrets.get("EMAIL_SENDER_NAME", "AURA System").strip()
OTP_EXPIRY_MINUTES = int(st.secrets.get("OTP_EXPIRY_MINUTES", 10))
ALLOW_LOCAL_OTP_FALLBACK = (
    str(st.secrets.get("ALLOW_LOCAL_OTP_FALLBACK", "false")).strip().lower()
    == "true"
)
RESEND_API_URL = "https://api.resend.com/emails"
MALAYSIA_TZ = ZoneInfo("Asia/Kuala_Lumpur")


def _normalize_staff_id_text(staff_id):
    text = str(staff_id or "").strip().lstrip("'")
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    return text


def current_malaysia_timestamp():
    return datetime.now(MALAYSIA_TZ).isoformat(sep=" ", timespec="seconds")


def audit_timestamp_to_malaysia(value):
    text = str(value or "").strip()
    if not text or text.lower() == "nan":
        return pd.NaT

    parsed = pd.to_datetime(text, errors="coerce")
    if pd.isna(parsed):
        return pd.NaT

    # Existing SQLite CURRENT_TIMESTAMP audit rows are UTC and have no timezone.
    # New audit rows include +08:00, so those are converted without shifting twice.
    if parsed.tzinfo is None:
        parsed = parsed.tz_localize("UTC")
    return parsed.tz_convert(MALAYSIA_TZ)


def format_audit_malaysia_time(value):
    parsed = audit_timestamp_to_malaysia(value)
    if pd.isna(parsed):
        return clean_display_value(value, "N/A")
    return parsed.strftime("%Y-%m-%d %H:%M:%S MYT")


@st.cache_resource
def setup_database():
    with sqlite3.connect(DB_FILE) as conn:
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
                questionnaire_id INTEGER,
                gpa REAL,
                stress_index REAL,
                prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                predicted_by TEXT,
                input_method TEXT,
                upload_batch_id TEXT,
                top_factors TEXT,
                interventions TEXT,
                ai_suggestions TEXT
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
            CREATE TABLE IF NOT EXISTS notification_logs (
                notification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_batch_id TEXT,
                prediction_id INTEGER,
                student_id TEXT,
                student_email TEXT,
                risk_level TEXT,
                notification_type TEXT,
                message_subject TEXT,
                message_body TEXT,
                send_status TEXT DEFAULT 'Pending',
                status_message TEXT,
                sent_by TEXT,
                sent_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS student_contacts (
                contact_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT UNIQUE NOT NULL,
                student_name TEXT,
                student_email TEXT,
                department TEXT,
                programme TEXT,
                phone_number TEXT,
                status TEXT DEFAULT 'active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS student_questionnaire (
                questionnaire_id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_email TEXT NOT NULL,
                student_id TEXT NOT NULL,
                semester TEXT,
                attendance_rate TEXT,
                assignment_status TEXT,
                coursework_mark TEXT,
                auto_tracking TEXT,
                performance_indicators TEXT,
                update_frequency TEXT,
                alert_types TEXT,
                suggestions_needed TEXT,
                alert_method TEXT,
                badge_motivation INTEGER,
                badge_achievements TEXT,
                reward_encouragement INTEGER,
                allow_risk_flagging TEXT,
                preferred_support TEXT,
                communication_method TEXT,
                communication_ease INTEGER,
                main_barrier TEXT,
                administration_support TEXT,
                high_risk_reminder_consent TEXT,
                preferred_contact_method TEXT,
                preferred_contact_details TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS student_course_study_plan (
                course_id INTEGER PRIMARY KEY AUTOINCREMENT,
                questionnaire_id INTEGER,
                student_email TEXT NOT NULL,
                student_id TEXT NOT NULL,
                course_name TEXT NOT NULL,
                course_credits REAL NOT NULL,
                weekly_study_hours REAL NOT NULL,
                recommended_weekly_hours REAL NOT NULL,
                study_gap REAL NOT NULL,
                status TEXT NOT NULL,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        course_failures.setup_course_tables(cursor)

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
                "upload_batch_id": "TEXT",
                "top_factors": "TEXT",
                "interventions": "TEXT",
                "ai_suggestions": "TEXT",
                "predicted_by": "TEXT",
                "questionnaire_id": "INTEGER",
            },
            "authorized_users": {
                "is_active": "INTEGER DEFAULT 1",
            },
            "student_questionnaire": {
                "semester": "TEXT",
                "high_risk_reminder_consent": "TEXT",
                "preferred_contact_method": "TEXT",
                "preferred_contact_details": "TEXT",
            },
            "student_contacts": {
                "programme": "TEXT",
                "phone_number": "TEXT",
                "status": "TEXT DEFAULT 'active'",
            },
        }
        for table, columns in migrations.items():
            cursor.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cursor.fetchall()}
            for column, column_type in columns.items():
                if column not in existing:
                    cursor.execute(
                        f"ALTER TABLE {table} ADD COLUMN {column} {column_type}"
                    )

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
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_history_batch "
            "ON prediction_history(upload_batch_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_prediction_history_questionnaire "
            "ON prediction_history(questionnaire_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_notification_logs_prediction "
            "ON notification_logs(prediction_id, send_status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_notification_logs_batch "
            "ON notification_logs(upload_batch_id, send_status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_student_contacts_student_id "
            "ON student_contacts(student_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_student_questionnaire_student_id "
            "ON student_questionnaire(student_id, submitted_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_student_questionnaire_semester "
            "ON student_questionnaire(student_id, semester, submitted_at)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_student_course_plan_student_id "
            "ON student_course_study_plan(student_id, submitted_at)"
        )

        if AUTHORIZED_USERS_FILE.exists():
            try:
                authorized_df = pd.read_csv(
                    AUTHORIZED_USERS_FILE,
                    dtype={"staff_id": str, "email": str, "is_active": str},
                    keep_default_na=False,
                ).fillna("")
                required_columns = {"staff_id", "email", "is_active"}
                if required_columns.issubset(set(authorized_df.columns)):
                    cursor.execute("DELETE FROM authorized_users")
                    for _, row in authorized_df.iterrows():
                        staff_id = _normalize_staff_id_text(row["staff_id"])
                        email = str(row["email"]).strip().lower()
                        role = str(row.get("role", "pending")).strip().lower() or "pending"
                        is_active = 1 if str(row["is_active"]).strip() != "0" else 0
                        name = str(row.get("name", staff_id)).strip() or staff_id
                        if not staff_id or not email:
                            continue
                        cursor.execute(
                            """
                            INSERT OR IGNORE INTO authorized_users
                                (name, email, staff_id, role, is_active, created_at)
                            VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (name, email, staff_id, role, is_active, app_now_text()),
                        )
            except Exception:
                pass

        if STUDENT_CONTACTS_FILE.exists():
            try:
                contacts_df = pd.read_csv(
                    STUDENT_CONTACTS_FILE,
                    dtype=str,
                    keep_default_na=False,
                ).fillna("")
                required_columns = {
                    "student_id",
                    "student_name",
                    "student_email",
                    "department",
                    "programme",
                    "phone_number",
                    "status",
                    "created_at",
                }
                if required_columns.issubset(set(contacts_df.columns)):
                    cursor.execute("DELETE FROM student_contacts")
                    for _, row in contacts_df.iterrows():
                        student_id = str(row["student_id"] or "").strip().upper()
                        if not student_id:
                            continue
                        cursor.execute(
                            """
                            INSERT OR REPLACE INTO student_contacts (
                                student_id, student_name, student_email,
                                department, programme, phone_number, status, created_at
                            )
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                student_id,
                                str(row.get("student_name", "")).strip(),
                                str(row.get("student_email", "")).strip().lower(),
                                str(row.get("department", "")).strip(),
                                str(row.get("programme", "")).strip(),
                                str(row.get("phone_number", "")).strip(),
                                str(row.get("status", "active")).strip() or "active",
                                str(row.get("created_at", "")).strip() or app_now_text(),
                            ),
                        )
            except Exception:
                pass

        if COURSE_CATALOG_FILE.exists():
            try:
                course_failures.sync_course_catalog_from_csv(conn, COURSE_CATALOG_FILE)
            except Exception:
                pass


def get_conn():
    conn = sqlite3.connect(DB_FILE, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA busy_timeout = 10000")
    return conn


setup_database()


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        st.error(f"Model file not found: {MODEL_FILE}")
        st.stop()
    return joblib.load(MODEL_FILE)


@st.cache_resource
def load_explainer(_model):
    try:
        import shap

        return shap.TreeExplainer(_model)
    except Exception:
        return None


def get_prediction_model():
    return load_model()


def get_model_features():
    return list(getattr(get_prediction_model(), "feature_names_in_", []))


def get_prediction_explainer():
    return load_explainer(get_prediction_model())


@st.cache_data(show_spinner=False)
def get_model_information_summary():
    summary = {
        "model_version": MODEL_FILE.stem.replace("best_randomforest_model_", "").upper(),
        "model_file": MODEL_FILE.name,
        "algorithm": "RandomForestClassifier",
        "training_dataset": MODEL_TRAINING_DATA_FILE.name,
        "training_rows": "Not available",
        "test_rows": "Not available",
        "dataset_rows": "Not available",
        "last_trained": "Not available",
        "performance": {},
        "error": "",
    }

    if MODEL_FILE.exists():
        modified = datetime.fromtimestamp(MODEL_FILE.stat().st_mtime)
        summary["last_trained"] = modified.strftime("%Y-%m-%d %H:%M:%S")

    if not MODEL_TRAINING_DATA_FILE.exists():
        summary["error"] = "Training dataset file was not found."
        return summary

    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split

        df = pd.read_csv(MODEL_TRAINING_DATA_FILE)
        summary["dataset_rows"] = len(df)

        missing_features = [
            feature for feature in MODEL_FEATURE_NAMES if feature not in df.columns
        ]
        if missing_features or "Dropout" not in df.columns:
            summary["error"] = "Training dataset is missing required model columns."
            return summary

        X = df[MODEL_FEATURE_NAMES]
        y = df["Dropout"].copy()
        flip_index = y.sample(frac=0.08, random_state=42).index
        y.loc[flip_index] = 1 - y.loc[flip_index]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y,
        )
        summary["training_rows"] = len(X_train)
        summary["test_rows"] = len(X_test)

        model = get_prediction_model()
        y_pred = model.predict(X_test)
        summary["performance"] = {
            "Accuracy": round(float(accuracy_score(y_test, y_pred)), 3),
            "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 3),
            "Recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 3),
            "F1 Score": round(float(f1_score(y_test, y_pred, zero_division=0)), 3),
        }

    except Exception as error:
        summary["error"] = str(error)

    return summary


def call_gemini_api(prompt):
    if not GEMINI_API_KEY:
        return False, (
            "Gemini API key is not configured. Add GEMINI_API_KEY to "
            ".streamlit/secrets.toml and restart the app."
        )

    api_url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODEL}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1200,
        },
    }
    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
            "User-Agent": "AURA-Streamlit/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            response_data = json.loads(response.read().decode("utf-8"))
        candidates = response_data.get("candidates") or []
        if not candidates:
            return False, "Gemini returned no response."
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "".join(
            str(part.get("text", ""))
            for part in parts
            if isinstance(part, dict)
        ).strip()
        if not text:
            return False, "Gemini returned an empty response."
        return True, text
    except urllib.error.HTTPError as error:
        try:
            error_data = json.loads(error.read().decode("utf-8"))
            message = error_data.get("error", {}).get("message", "")
        except Exception:
            message = ""
        if error.code in {400, 401, 403}:
            return False, (
                message
                or "Gemini rejected the API key or its permissions."
            )
        if error.code == 429:
            return False, "Gemini API quota or rate limit was reached."
        return False, message or f"Gemini API request failed with HTTP {error.code}."
    except Exception as error:
        return False, f"Gemini API connection failed: {type(error).__name__}."


# =========================
# ORIGINAL CHATBOT CODE
# =========================
API_KEY =  st.secrets.get("ZHIPUAI_API_KEY", "")
MODEL = "glm-4-flash"
client = (
    ZhipuAI(api_key=API_KEY, timeout=15.0, max_retries=1)
    if API_KEY
    else None
)

SYSTEM_PROMPT = """你是一个专业的学生辍学风险分析助手。
你的职责是：
- 根据学生的学习情况、出勤率、家庭背景等信息评估辍学风险
- 用温和、关怀的语气与老师或教育工作者沟通
- 给出具体的干预建议，帮助高风险学生留在学校
- 回答要简洁专业，避免过于学术化的术语
如果用户提供了学生数据，请给出风险等级（低/中/高）和建议。用简短的英文"""

QUICK_QUESTIONS = [
    "What to do if GPA is low?",
    "What does a high or low dropout rate represent?",
    "Given my current situation, what suggestions do you have?",
]

def local_chat_reply(message):
    text = str(message or "").strip().lower()
    if "gpa" in text:
        return (
            "A low GPA should trigger a supportive review, not an automatic conclusion. "
            "Check the affected courses, attendance, assessment deadlines, and study load. "
            "Agree on two or three measurable actions such as weekly tutoring, assignment "
            "milestones, and an advisor check-in within seven days."
        )
    if "high" in text and ("risk" in text or "dropout" in text):
        return (
            "High risk means the model found several patterns associated with withdrawal. "
            "It is an early-warning signal, not a certainty. Review the student's context, "
            "contact them promptly, and record a short support plan with a follow-up date."
        )
    if "low" in text and ("risk" in text or "dropout" in text):
        return (
            "Low risk means the current indicators appear comparatively stable. Continue "
            "normal monitoring and avoid treating the result as a guarantee; circumstances "
            "can change between assessments."
        )
    return (
        "Start with a brief, private conversation to understand the student's academic and "
        "personal context. Review attendance, GPA, assignment delays, study load, and stress. "
        "Choose practical support actions, assign an owner, and set a follow-up date so progress "
        "can be reviewed."
    )


def get_ai_chat_reply(messages, prefer_live=True):
    if prefer_live and client is not None:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, *messages],
                stream=False,
            )
            reply = response.choices[0].message.content
            if str(reply or "").strip():
                return str(reply).strip(), True
        except Exception:
            pass
    latest_message = messages[-1]["content"] if messages else ""
    return local_chat_reply(latest_message), False


def chatbot_page():

    st.markdown("""
    <style>
    .stChatMessage {
        background-color: #1e2a3a !important;
        border-radius: 12px !important;
        border: 1px solid #3d4f6e !important;
    }
    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage ol,
    .stChatMessage ul,
    .stChatMessage span,
    .stChatMessage div {
        color: #e8eaf6 !important;
        font-size: 15px !important;
    }
    div.st-key-quick_0 button,
    div.st-key-quick_1 button,
    div.st-key-quick_2 button {
        min-height: 3rem;
        border-radius: 999px;
        color: white !important;
        font-weight: 800;
        line-height: 1.2;
        white-space: normal;
        border-width: 1px;
        box-shadow: 0 10px 22px rgba(0, 0, 0, 0.22);
        transition: transform 0.15s ease, filter 0.15s ease, box-shadow 0.15s ease;
    }
    div.st-key-quick_0 button p,
    div.st-key-quick_1 button p,
    div.st-key-quick_2 button p {
        color: white !important;
        font-weight: 800;
    }
    div.st-key-quick_0 button {
        background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;
        border-color: #60a5fa !important;
    }
    div.st-key-quick_1 button {
        background: linear-gradient(135deg, #f59e0b, #d97706) !important;
        border-color: #fbbf24 !important;
    }
    div.st-key-quick_2 button {
        background: linear-gradient(135deg, #16a34a, #15803d) !important;
        border-color: #4ade80 !important;
    }
    div.st-key-quick_0 button:hover,
    div.st-key-quick_1 button:hover,
    div.st-key-quick_2 button:hover {
        filter: brightness(1.08);
        transform: translateY(-1px);
        box-shadow: 0 14px 26px rgba(0, 0, 0, 0.28);
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AURA AI Assistant")
    st.caption(
        "Ask about student-risk interpretation, intervention planning, and advisor follow-up."
    )

    if st.button("← Back to Dashboard"):
        st.session_state.sidebar_selected_page = "Dashboard"
        st.session_state.page = "dashboard"
        st.rerun()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write("**Quick Questions:**")
    cols = st.columns(3)
    for i, q in enumerate(QUICK_QUESTIONS):
        with cols[i]:
            if st.button(q, use_container_width=True, key=f"quick_{i}", type="primary"):
                st.session_state.chat_history.append({"role": "user", "content": q})
                reply, _ = get_ai_chat_reply(
                    st.session_state.chat_history,
                    prefer_live=False,
                )
                st.session_state.chat_history.append({"role": "assistant", "content": reply})
                st.rerun()

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            reply, live_reply = get_ai_chat_reply(st.session_state.chat_history)
            st.write(reply)
            if not live_reply:
                st.caption("AURA used its built-in advisor guidance because live AI was unavailable.")
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()


def show_chat_fab(key_suffix="default"):
    st.markdown("""
        <style>
        div[class*="st-key-fab_chat"] button {
            min-height: 3.1rem;
            border-radius: 999px;
            border: 1px solid #fde047 !important;
            background: linear-gradient(135deg, #ff2d55, #ffb703) !important;
            color: #ffffff !important;
            font-size: 1rem;
            font-weight: 900;
            box-shadow:
                0 0 0 3px rgba(255, 183, 3, 0.18),
                0 12px 30px rgba(255, 45, 85, 0.35),
                0 0 24px rgba(255, 183, 3, 0.45);
            transition: transform 0.15s ease, filter 0.15s ease, box-shadow 0.15s ease;
        }
        div[class*="st-key-fab_chat"] button p {
            color: #ffffff !important;
            font-weight: 900;
        }
        div[class*="st-key-fab_chat"] button:hover {
            filter: brightness(1.12);
            transform: translateY(-1px);
            box-shadow:
                0 0 0 4px rgba(255, 183, 3, 0.24),
                0 16px 34px rgba(255, 45, 85, 0.42),
                0 0 30px rgba(255, 183, 3, 0.58);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("---")
    col1, col2, col3 = st.columns([4, 1.4, 0.1])

    with col2:
        if st.button("💬 AI Assistant", key=f"fab_chat_{key_suffix}", use_container_width=True, type="primary"):
            st.session_state.page = "chatbot"
            st.rerun()


st.markdown(
    """
    <style>
    .stApp { background-color: #0f1117; color: white; }
    section[data-testid="stSidebar"] {
        background: #111722;
        border-right: 1px solid #2b3445;
    }
    section[data-testid="stSidebar"] [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        text-align: center;
        width: 100%;
        margin-bottom: 0.6rem;
    }
    section[data-testid="stSidebar"] [data-testid="stImage"] img {
        margin-left: auto;
        margin-right: auto;
        display: block;
    }
    .sidebar-logo-wrap {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0.25rem 0 1.25rem;
    }
    .sidebar-logo-wrap img {
        width: 112px;
        height: 112px;
        object-fit: contain;
        border-radius: 8px;
        background: white;
        display: block;
    }
    .sidebar-user-card {
        border: 1px solid #303846;
        border-radius: 8px;
        background: #141923;
        padding: 0.8rem 0.9rem;
        margin: 0.4rem 0 1rem;
    }
    .sidebar-user-label {
        color: #9ca3af;
        font-size: 0.72rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 0.2rem;
    }
    .sidebar-user-value {
        color: #f8fafc;
        font-size: 0.9rem;
        font-weight: 750;
        overflow-wrap: anywhere;
        margin-bottom: 0.55rem;
    }
    .sidebar-section-title {
        color: #f4c542;
        font-size: 0.76rem;
        font-weight: 900;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin: 0.7rem 0 0.25rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        border-radius: 8px;
        min-height: 2.45rem;
        justify-content: flex-start;
        font-weight: 750;
        border: 1px solid #303846;
        background: #141923;
        color: #f8fafc;
        transition: background 0.15s ease, border-color 0.15s ease, transform 0.15s ease;
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        background: #1b2535;
        border-color: #3d4f6e;
        transform: translateX(2px);
    }
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #1f77ff, #243f91);
        border-color: #5aa2ff;
        color: white;
        box-shadow: 0 8px 22px rgba(31, 119, 255, 0.2);
    }
    .aura-back-top {
        position: fixed;
        right: 24px;
        bottom: 24px;
        z-index: 9999;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 46px;
        border-radius: 50%;
        border: 1px solid #5aa2ff;
        background: linear-gradient(135deg, #1f77ff, #243f91);
        color: white !important;
        font-size: 1.35rem;
        line-height: 1;
        text-decoration: none !important;
        box-shadow: 0 12px 28px rgba(31, 119, 255, 0.28);
    }
    .aura-back-top:hover {
        border-color: #93c5fd;
        filter: brightness(1.08);
    }
    .aura-header {
        background: linear-gradient(90deg, #243f91, #1f77ff);
        color: white;
        padding: 16px 22px;
        border-radius: 14px;
        margin-bottom: 20px;
    }
    .home-hero {
        border: 1px solid #2b3445;
        border-radius: 12px;
        background:
            linear-gradient(135deg, rgba(31,119,255,0.18), rgba(20,25,35,0.95)),
            #141923;
        padding: 2.1rem;
        margin: 2rem auto 1.25rem;
        display: grid;
        grid-template-columns: minmax(170px, 260px) minmax(0, 1fr);
        gap: 2rem;
        align-items: center;
        box-shadow: 0 18px 55px rgba(0, 0, 0, 0.24);
    }
    .home-logo-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 1.25rem;
        display: flex;
        align-items: center;
        justify-content: center;
        min-height: 220px;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .home-logo-card img {
        width: 100%;
        max-height: 210px;
        object-fit: contain;
    }
    .home-logo-fallback {
        color: #243f91;
        font-size: 2.5rem;
        font-weight: 900;
    }
    .home-kicker {
        color: #f4c542;
        font-size: 0.84rem;
        font-weight: 850;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.65rem;
    }
    .home-copy h1 {
        color: #f8fafc;
        font-size: clamp(2.4rem, 4vw, 4.5rem);
        line-height: 1;
        margin: 0 0 0.8rem;
        letter-spacing: 0;
    }
    .home-copy p {
        color: #cbd5e1;
        font-size: 1.08rem;
        line-height: 1.65;
        max-width: 760px;
        margin: 0;
    }
    .home-stat-row {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin-top: 1.35rem;
    }
    .home-stat {
        border: 1px solid #303846;
        border-radius: 8px;
        background: rgba(15, 17, 23, 0.68);
        padding: 0.9rem 1rem;
    }
    .home-stat strong {
        display: block;
        color: #f8fafc;
        font-size: 1rem;
        margin-bottom: 0.2rem;
    }
    .home-stat span {
        color: #9ca3af;
        font-size: 0.88rem;
        font-weight: 650;
    }
    .home-action-title {
        color: #f8fafc;
        font-weight: 850;
        font-size: 1.05rem;
        margin: 1.2rem 0 0.55rem;
    }
    .home-feature-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.85rem;
        margin-top: 1.1rem;
    }
    .home-feature {
        border: 1px solid #303846;
        border-radius: 8px;
        background: #141923;
        padding: 1rem 1.1rem;
        min-height: 7rem;
    }
    .home-feature h3 {
        color: #f8fafc;
        font-size: 1rem;
        margin: 0 0 0.45rem;
    }
    .home-feature p {
        color: #a7adbb;
        font-size: 0.9rem;
        line-height: 1.5;
        margin: 0;
    }
    .risk-card {
        padding: 18px;
        border-radius: 12px;
        color: white;
        margin-top: 12px;
    }
    .student-list-meta {
        color: #a7adbb;
        font-size: 0.9rem;
        margin-top: -0.2rem;
        margin-bottom: 1rem;
    }
    .student-header-cell {
        color: #f8fafc;
        font-size: 0.95rem;
        font-weight: 800;
        padding: 0.6rem 0 0.8rem;
        border-bottom: 1px solid #303846;
    }
    .student-row-cell {
        min-height: 4.4rem;
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding: 0.25rem 0;
    }
    .student-row-primary {
        color: #f8fafc;
        font-weight: 800;
        font-size: 1rem;
        line-height: 1.25;
    }
    .student-row-subtle {
        color: #9ca3af;
        font-size: 0.84rem;
        font-weight: 600;
        margin-top: 0.35rem;
    }
    .student-probability {
        color: #f8fafc;
        font-size: 1.02rem;
        font-weight: 800;
        min-height: 4.4rem;
        display: flex;
        align-items: center;
    }
    .student-risk-badge {
        color: white;
        border-radius: 8px;
        min-height: 2.55rem;
        width: 100%;
        max-width: 14rem;
        padding: 0.45rem 0.8rem;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        font-weight: 800;
        box-shadow: 0 8px 22px rgba(0, 0, 0, 0.18);
    }
    .questionnaire-badge {
        border-radius: 999px;
        padding: 0.42rem 0.75rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 8.5rem;
        font-size: 0.82rem;
        font-weight: 850;
        border: 1px solid transparent;
    }
    .questionnaire-completed {
        background: rgba(22, 163, 74, 0.16);
        color: #86efac;
        border-color: rgba(34, 197, 94, 0.35);
    }
    .questionnaire-missing {
        background: rgba(245, 158, 11, 0.14);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.35);
    }
    .notification-badge {
        border-radius: 999px;
        padding: 0.42rem 0.75rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 7.5rem;
        font-size: 0.82rem;
        font-weight: 850;
        border: 1px solid transparent;
    }
    .notification-not-required {
        background: rgba(148, 163, 184, 0.12);
        color: #cbd5e1;
        border-color: rgba(148, 163, 184, 0.28);
    }
    .notification-pending {
        background: rgba(245, 158, 11, 0.14);
        color: #fbbf24;
        border-color: rgba(245, 158, 11, 0.35);
    }
    .notification-sent,
    .notification-resent {
        background: rgba(22, 163, 74, 0.16);
        color: #86efac;
        border-color: rgba(34, 197, 94, 0.35);
    }
    .notification-failed {
        background: rgba(239, 68, 68, 0.16);
        color: #fca5a5;
        border-color: rgba(239, 68, 68, 0.35);
    }
    .student-row-divider {
        border-bottom: 1px solid #303846;
        margin: 0.55rem 0 0.85rem;
    }
    .pager-label {
        text-align: center;
        font-weight: 800;
        padding-top: 0.45rem;
        color: #f8fafc;
    }
    .detail-hero {
        border: 1px solid #303846;
        border-radius: 10px;
        padding: 1.25rem 1.4rem;
        margin: 0.8rem 0 1rem;
        background: #141923;
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 1rem;
    }
    .detail-hero h2 {
        margin: 0.15rem 0;
        color: #f8fafc;
    }
    .detail-eyebrow {
        color: #9ca3af;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.06rem;
    }
    .detail-subtitle {
        color: #cbd5e1;
        margin-top: 0.25rem;
        font-weight: 600;
    }
    .detail-status {
        color: white;
        border-radius: 10px;
        padding: 0.75rem 1.1rem;
        text-align: center;
        min-width: 11rem;
        font-weight: 800;
    }
    .detail-status span {
        display: block;
        font-size: 0.82rem;
        font-weight: 600;
        opacity: 0.92;
        margin-top: 0.2rem;
    }
    .detail-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.75rem;
        margin: 0.8rem 0 1rem;
    }
    .detail-item {
        border: 1px solid #303846;
        border-radius: 8px;
        background: #141923;
        padding: 0.85rem 0.95rem;
        min-height: 4.8rem;
    }
    .detail-label {
        color: #9ca3af;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }
    .detail-value {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 750;
        overflow-wrap: anywhere;
    }
    .detail-list-card {
        border: 1px solid #303846;
        border-radius: 8px;
        background: #141923;
        padding: 0.9rem 1.1rem;
        margin-top: 0.75rem;
    }
    [data-testid="stHorizontalBlock"] {
        align-items: flex-start;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        min-width: 0;
    }

    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] > div,
    [data-testid="stHorizontalBlock"] [data-testid="stTextInput"],
    [data-testid="stHorizontalBlock"] [data-testid="stNumberInput"],
    [data-testid="stHorizontalBlock"] [data-testid="stSelectbox"],
    [data-testid="stHorizontalBlock"] [data-testid="stMultiSelect"],
    [data-testid="stHorizontalBlock"] [data-testid="stTextArea"],
    [data-testid="stHorizontalBlock"] [data-testid="stDateInput"],
    [data-testid="stHorizontalBlock"] [data-testid="stTimeInput"],
    [data-testid="stHorizontalBlock"] [data-testid="stFileUploader"] {
        width: 100%;
    }

    [data-testid="stHorizontalBlock"] [data-testid="stButton"] > button,
    [data-testid="stHorizontalBlock"] [data-testid="stDownloadButton"] > button {
        width: 100%;
    }

    [data-testid="stTabs"] [role="tablist"] {
        align-items: stretch;
        gap: 0.2rem;
    }

    [data-testid="stTabs"] [role="tabpanel"],
    [data-testid="stMainBlockContainer"] > div,
    .block-container > div {
        width: 100%;
    }

    @media (max-width: 900px) {
        .detail-hero {
            display: block;
        }
        .detail-status {
            margin-top: 1rem;
            width: 100%;
        }
        .detail-grid {
            grid-template-columns: 1fr;
        }
        .home-hero {
            grid-template-columns: 1fr;
            padding: 1.25rem;
            margin-top: 1rem;
        }
        .home-logo-card {
            min-height: 180px;
        }
        .home-stat-row,
        .home-feature-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    :root {
        --aura-canvas: #f4f3ee;
        --aura-surface: #ffffff;
        --aura-surface-soft: #faf9f6;
        --aura-ink: #172033;
        --aura-muted: #687083;
        --aura-line: #deded8;
        --aura-line-strong: #cfd1d8;
        --aura-navy: #111d31;
        --aura-navy-soft: #1a2a44;
        --aura-indigo: #665cf6;
        --aura-indigo-dark: #5147df;
        --aura-indigo-soft: #eeecff;
        --aura-teal: #2aa89a;
        --aura-gold: #d7a747;
        --aura-danger: #d94d62;
        --aura-shadow-sm: 0 8px 24px rgba(24, 32, 51, 0.06);
        --aura-shadow-md: 0 20px 55px rgba(24, 32, 51, 0.10);
        --aura-radius-sm: 12px;
        --aura-radius-md: 18px;
        --aura-radius-lg: 28px;
        color-scheme: light;
    }

    html,
    body,
    [class*="css"] {
        font-family: "Aptos", "Segoe UI Variable", "Segoe UI", Inter, sans-serif;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at 82% 2%, rgba(102, 92, 246, 0.10), transparent 28rem),
            radial-gradient(circle at 16% 88%, rgba(42, 168, 154, 0.08), transparent 30rem),
            var(--aura-canvas) !important;
        color: var(--aura-ink) !important;
    }

    header[data-testid="stHeader"] {
        background: rgba(244, 243, 238, 0.88) !important;
        border-bottom: 1px solid rgba(23, 32, 51, 0.06);
        backdrop-filter: blur(16px);
    }

    [data-testid="stToolbar"] {
        color: var(--aura-ink) !important;
    }

    [data-testid="stMainBlockContainer"],
    .block-container {
        max-width: 1380px;
        padding: 3.2rem 3rem 5rem;
    }

    h1, h2, h3, h4, h5, h6,
    [data-testid="stHeadingWithActionElements"] {
        color: var(--aura-ink) !important;
        letter-spacing: -0.025em;
    }

    /* Remove Streamlit's automatic heading permalink icon everywhere. */
    [data-testid="stHeaderActionElements"],
    [data-testid="stHeadingWithActionElements"] > a,
    [data-testid="stHeadingWithActionElements"] a[href^="#"],
    h1 > a[href^="#"],
    h2 > a[href^="#"],
    h3 > a[href^="#"],
    h4 > a[href^="#"],
    h5 > a[href^="#"],
    h6 > a[href^="#"] {
        display: none !important;
    }

    h1 {
        font-size: clamp(2.15rem, 3vw, 3.15rem) !important;
        line-height: 1.04 !important;
        font-weight: 760 !important;
        margin-bottom: 0.85rem !important;
    }

    h1::before {
        content: "AURA  /  WORKSPACE";
        display: block;
        margin-bottom: 0.72rem;
        color: var(--aura-indigo);
        font-size: 0.72rem;
        line-height: 1;
        font-weight: 800;
        letter-spacing: 0.15em;
    }

    h2 {
        font-size: 1.55rem !important;
        font-weight: 740 !important;
    }

    h3 {
        font-size: 1.12rem !important;
        font-weight: 720 !important;
    }

    p,
    label,
    [data-testid="stCaptionContainer"],
    [data-testid="stMarkdownContainer"],
    [data-testid="stWidgetLabel"] {
        color: var(--aura-ink);
    }

    [data-testid="stCaptionContainer"],
    small {
        color: var(--aura-muted) !important;
    }

    hr {
        border-color: var(--aura-line) !important;
        margin: 1.75rem 0 !important;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    button[data-testid="baseButton-secondary"],
    button[data-testid="baseButton-primary"] {
        min-height: 2.85rem;
        border-radius: 13px !important;
        border: 1px solid var(--aura-line-strong) !important;
        background: rgba(255, 255, 255, 0.94) !important;
        color: var(--aura-ink) !important;
        font-weight: 720 !important;
        letter-spacing: -0.01em;
        box-shadow: 0 2px 0 rgba(23, 32, 51, 0.02);
        transition:
            transform 0.18s ease,
            box-shadow 0.18s ease,
            border-color 0.18s ease,
            background 0.18s ease !important;
    }

    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button,
    div[data-testid="stButton"] > button *,
    div[data-testid="stDownloadButton"] > button * {
        max-width: 100% !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.25 !important;
    }

    div[data-testid="stButton"] > button p,
    div[data-testid="stDownloadButton"] > button p,
    button[data-testid="baseButton-secondary"] p,
    button[data-testid="baseButton-primary"] p {
        color: inherit !important;
        font-weight: inherit !important;
    }

    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover,
    button[data-testid="baseButton-secondary"]:hover {
        transform: translateY(-1px);
        border-color: #aaa6e9 !important;
        background: #ffffff !important;
        color: var(--aura-indigo-dark) !important;
        box-shadow: 0 10px 25px rgba(50, 47, 106, 0.10);
    }

    div[data-testid="stButton"] > button[kind="primary"],
    button[data-testid="baseButton-primary"] {
        border-color: var(--aura-indigo) !important;
        background: linear-gradient(135deg, #7167ff, #5950e6) !important;
        color: #ffffff !important;
        box-shadow: 0 10px 26px rgba(102, 92, 246, 0.24);
    }

    div[data-testid="stButton"] > button[kind="primary"]:hover,
    button[data-testid="baseButton-primary"]:hover {
        border-color: var(--aura-indigo-dark) !important;
        background: linear-gradient(135deg, #675df5, #4f45dc) !important;
        color: #ffffff !important;
        box-shadow: 0 14px 32px rgba(102, 92, 246, 0.30);
    }

    div[data-testid="stButton"] > button:disabled,
    button:disabled {
        opacity: 0.48 !important;
        transform: none !important;
        box-shadow: none !important;
    }

    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="textarea"],
    [data-testid="stNumberInputContainer"],
    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea {
        border-color: var(--aura-line-strong) !important;
        border-radius: 13px !important;
        background: #ffffff !important;
        color: var(--aura-ink) !important;
        box-shadow: 0 1px 0 rgba(23, 32, 51, 0.02);
    }

    [data-testid="stSelectbox"] div[data-baseweb="select"] *,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] *,
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stMultiSelect"] [role="combobox"],
    [role="option"] {
        max-width: 100% !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
    }

    [data-testid="stTextInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="input"],
    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stTimeInput"] div[data-baseweb="input"],
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div,
    [data-testid="stTextArea"] div[data-baseweb="textarea"],
    [data-testid="stTextArea"] textarea {
        min-height: 3.15rem;
        border: 1.5px solid #aeb4c3 !important;
        border-radius: 13px !important;
        background: #f8f9fc !important;
        box-shadow:
            inset 0 1px 0 rgba(255, 255, 255, 0.88),
            0 2px 6px rgba(23, 32, 51, 0.045) !important;
    }

    [data-testid="stTextInput"] div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] div[data-baseweb="input"] > div,
    [data-testid="stDateInput"] div[data-baseweb="input"] > div,
    [data-testid="stTimeInput"] div[data-baseweb="input"] > div {
        border: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input,
    [data-testid="stTextArea"] textarea {
        background: transparent !important;
    }

    [data-testid="stNumberInputContainer"] {
        min-height: 3.15rem !important;
        height: 3.15rem !important;
        overflow: hidden !important;
    }

    [data-testid="stNumberInputContainer"] button {
        min-height: 3.15rem !important;
        height: 3.15rem !important;
        border-radius: 0 !important;
    }

    [data-testid="stTextInput"] div[data-baseweb="input"]:hover,
    [data-testid="stNumberInput"] div[data-baseweb="input"]:hover,
    [data-testid="stDateInput"] div[data-baseweb="input"]:hover,
    [data-testid="stTimeInput"] div[data-baseweb="input"]:hover,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div:hover,
    [data-testid="stTextArea"] div[data-baseweb="textarea"]:hover {
        border-color: #858da0 !important;
        background: #ffffff !important;
    }

    [data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
    [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
    [data-testid="stDateInput"] div[data-baseweb="input"]:focus-within,
    [data-testid="stTimeInput"] div[data-baseweb="input"]:focus-within,
    [data-testid="stSelectbox"] div[data-baseweb="select"] > div:focus-within,
    [data-testid="stMultiSelect"] div[data-baseweb="select"] > div:focus-within,
    [data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within {
        border-color: var(--aura-indigo) !important;
        background: #ffffff !important;
        box-shadow:
            0 0 0 3px rgba(102, 92, 246, 0.15),
            0 5px 15px rgba(23, 32, 51, 0.08) !important;
    }

    [data-testid="stFileUploader"] {
        padding: 0.2rem;
        border-radius: 18px;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        padding: 0.8rem 0.35rem 0.55rem;
        border: 1px solid #c5c9d3;
        border-radius: 13px;
        background: #f8f9fc;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stNumberInput"] input,
    div[data-baseweb="select"] * {
        color: var(--aura-ink) !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #9a9fad !important;
    }

    div[data-baseweb="input"] > div:focus-within,
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"]:focus-within {
        border-color: var(--aura-indigo) !important;
        box-shadow: 0 0 0 3px rgba(102, 92, 246, 0.13) !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {
        border: 1px solid var(--aura-line) !important;
        border-radius: 14px !important;
        background: #ffffff !important;
        color: var(--aura-ink) !important;
        box-shadow: var(--aura-shadow-md) !important;
    }

    [role="option"] {
        color: var(--aura-ink) !important;
    }

    [role="option"]:hover {
        background: var(--aura-indigo-soft) !important;
    }

    [data-testid="stRadio"] label,
    [data-testid="stCheckbox"] label,
    [data-testid="stMultiSelect"] label,
    [data-testid="stSelectbox"] label,
    [data-testid="stFileUploader"] label {
        color: var(--aura-ink) !important;
    }

    [data-testid="stMetric"] {
        position: relative;
        overflow: hidden;
        min-height: 9.2rem;
        padding: 1.25rem 1.3rem 1.15rem;
        border: 1px solid rgba(23, 32, 51, 0.09);
        border-radius: var(--aura-radius-md);
        background:
            linear-gradient(145deg, rgba(255,255,255,0.98), rgba(250,249,246,0.98));
        box-shadow: var(--aura-shadow-sm);
    }

    [data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 5rem;
        height: 5rem;
        border-radius: 0 0 0 5rem;
        background: linear-gradient(135deg, rgba(102, 92, 246, 0.11), rgba(42, 168, 154, 0.08));
    }

    [data-testid="stMetricLabel"] {
        color: var(--aura-muted) !important;
        font-size: 0.79rem !important;
        font-weight: 760 !important;
        letter-spacing: 0.045em;
        text-transform: uppercase;
    }

    [data-testid="stMetricValue"] {
        color: var(--aura-ink) !important;
        font-size: clamp(1.65rem, 2.2vw, 2.35rem) !important;
        font-weight: 780 !important;
        letter-spacing: -0.05em;
    }

    [data-testid="stMetricDelta"] {
        color: var(--aura-teal) !important;
    }

    [data-testid="stPlotlyChart"] {
        overflow: hidden;
        margin: 0.8rem 0 1.3rem;
        padding: 1rem 1rem 0.45rem;
        border: 1px solid rgba(23, 32, 51, 0.09);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: var(--aura-shadow-sm);
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"] {
        overflow: hidden;
        border: 1px solid rgba(23, 32, 51, 0.10) !important;
        border-radius: 16px !important;
        background: #ffffff;
        box-shadow: var(--aura-shadow-sm);
    }

    [data-testid="stFileUploader"] section {
        border: 1.5px dashed #aaa6e9 !important;
        border-radius: 18px !important;
        background:
            linear-gradient(135deg, rgba(238, 236, 255, 0.72), rgba(255, 255, 255, 0.92)) !important;
        padding: 1.2rem !important;
    }

    [data-testid="stFileUploader"] section svg {
        color: var(--aura-indigo) !important;
    }

    [data-testid="stExpander"] {
        overflow: hidden;
        border: 1px solid var(--aura-line) !important;
        border-radius: 15px !important;
        background: rgba(255, 255, 255, 0.88) !important;
        box-shadow: 0 5px 18px rgba(23, 32, 51, 0.04);
    }

    button[data-baseweb="tab"] {
        min-height: 2.9rem;
        margin-right: 0.35rem;
        padding: 0.55rem 1rem !important;
        border-radius: 12px 12px 0 0 !important;
        color: var(--aura-muted) !important;
        font-weight: 720 !important;
    }

    button[data-baseweb="tab"],
    button[data-baseweb="tab"] * {
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        line-height: 1.2 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        background: #ffffff !important;
        color: var(--aura-indigo-dark) !important;
    }

    [data-baseweb="tab-highlight"] {
        background-color: var(--aura-indigo) !important;
    }

    [data-baseweb="tab-border"] {
        background-color: var(--aura-line) !important;
    }

    [data-testid="stAlert"] {
        border: 1px solid rgba(23, 32, 51, 0.08) !important;
        border-radius: 15px !important;
        color: var(--aura-ink) !important;
        box-shadow: 0 6px 20px rgba(23, 32, 51, 0.04);
    }

    [data-testid="stAlert"] p {
        color: var(--aura-ink) !important;
    }

    section[data-testid="stSidebar"] {
        width: 20.5rem !important;
        min-width: 20.5rem !important;
        background:
            radial-gradient(circle at 25% 0%, rgba(102, 92, 246, 0.33), transparent 16rem),
            linear-gradient(180deg, #15243b 0%, #101a2b 55%, #0d1625 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 14px 0 45px rgba(17, 29, 49, 0.11);
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding: 1.05rem 0.9rem 1.4rem;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #f7f8fc !important;
    }

    .sidebar-logo-wrap {
        width: 100%;
        display: grid;
        grid-template-columns: 58px minmax(0, 1fr);
        gap: 0.85rem;
        align-items: center;
        justify-content: stretch;
        margin: 0.15rem 0 1rem;
        padding: 0.2rem 0.25rem;
    }

    .sidebar-logo-wrap img {
        width: 58px;
        height: 58px;
        margin: 0;
        padding: 0.3rem;
        object-fit: contain;
        border-radius: 16px;
        background: #ffffff;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.20);
    }

    .sidebar-brand-name {
        color: #ffffff;
        font-size: 1.22rem;
        font-weight: 800;
        line-height: 1.05;
        letter-spacing: -0.035em;
    }

    .sidebar-brand-subtitle {
        margin-top: 0.32rem;
        color: #aebbd0;
        font-size: 0.69rem;
        font-weight: 720;
        line-height: 1.25;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .sidebar-user-card {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 17px;
        background: rgba(255, 255, 255, 0.065);
        padding: 0.9rem 0.95rem;
        margin: 0.35rem 0 1rem;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }

    .sidebar-user-card::before {
        content: "●  SECURE SESSION";
        display: block;
        margin-bottom: 0.72rem;
        color: #73dbc9;
        font-size: 0.62rem;
        font-weight: 820;
        letter-spacing: 0.11em;
    }

    .sidebar-user-label {
        color: #91a0ba !important;
        font-size: 0.64rem;
        font-weight: 800;
        letter-spacing: 0.09em;
    }

    .sidebar-user-value {
        color: #f8f9fd !important;
        font-size: 0.84rem;
        font-weight: 680;
    }

    .sidebar-section-title {
        color: #9eaacc;
        font-size: 0.66rem;
        font-weight: 820;
        letter-spacing: 0.13em;
        margin: 1rem 0 0.4rem;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button {
        min-height: 3rem;
        height: auto !important;
        margin: 0.08rem 0;
        padding: 0.65rem 0.9rem;
        justify-content: center;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 12px !important;
        background: rgba(255, 255, 255, 0.055) !important;
        color: #e5eaf3 !important;
        font-size: 0.86rem;
        font-weight: 690 !important;
        box-shadow: none !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button span,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button div {
        color: #f7f9fc !important;
        -webkit-text-fill-color: #f7f9fc !important;
        font-weight: inherit !important;
        opacity: 1 !important;
        width: 100% !important;
        max-width: 100% !important;
        overflow: visible !important;
        text-overflow: clip !important;
        white-space: normal !important;
        overflow-wrap: anywhere !important;
        word-break: normal !important;
        line-height: 1.28 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button p {
        display: block !important;
        width: 100% !important;
        margin: 0 !important;
        text-align: center !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {
        transform: none;
        border-color: rgba(171, 164, 255, 0.48) !important;
        background: rgba(255, 255, 255, 0.12) !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover span,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] {
        border-color: #9189ff !important;
        background: linear-gradient(135deg, #7167ff, #554bdc) !important;
        color: #ffffff !important;
        font-weight: 790 !important;
        box-shadow:
            inset 4px 0 0 #c6c2ff,
            0 9px 22px rgba(5, 9, 20, 0.24) !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] *,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] p,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] span,
    section[data-testid="stSidebar"] div[data-testid="stButton"] button[kind="primary"] div {
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[data-testid="stBaseButton-headerNoPadding"],
    button[data-testid="stSidebarCollapseButton"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] div[class*="st-key-nav_"] button::after {
        display: none !important;
        content: none !important;
    }

    .home-hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 30px;
        background:
            radial-gradient(circle at 86% 10%, rgba(105, 95, 255, 0.38), transparent 22rem),
            radial-gradient(circle at 4% 100%, rgba(42, 168, 154, 0.18), transparent 24rem),
            linear-gradient(145deg, #172943, #101b2d 62%, #0d1727);
        padding: clamp(1.5rem, 4vw, 3.3rem);
        margin: 0.7rem auto 1.6rem;
        grid-template-columns: minmax(190px, 250px) minmax(0, 1fr);
        gap: clamp(1.5rem, 4vw, 3.4rem);
        box-shadow: 0 28px 75px rgba(17, 29, 49, 0.22);
    }

    .home-hero::after {
        content: "";
        position: absolute;
        right: -5rem;
        bottom: -7rem;
        width: 18rem;
        height: 18rem;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 50%;
        box-shadow:
            0 0 0 3rem rgba(255, 255, 255, 0.025),
            0 0 0 6rem rgba(255, 255, 255, 0.018);
        pointer-events: none;
    }

    .home-logo-card {
        position: relative;
        z-index: 1;
        min-height: 230px;
        padding: 1.25rem;
        border: 1px solid rgba(255, 255, 255, 0.70);
        border-radius: 24px;
        background:
            linear-gradient(145deg, rgba(255,255,255,1), rgba(242,243,248,0.96));
        box-shadow:
            0 22px 45px rgba(0, 0, 0, 0.20),
            inset 0 1px 0 #ffffff;
        transform: rotate(-1.5deg);
    }

    .home-logo-card img {
        max-height: 205px;
        filter: saturate(0.94) contrast(1.03);
    }

    .home-copy {
        position: relative;
        z-index: 1;
    }

    .home-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        width: fit-content;
        margin-bottom: 1rem;
        padding: 0.48rem 0.72rem;
        border: 1px solid rgba(151, 143, 255, 0.30);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.075);
        color: #c9c5ff;
        font-size: 0.69rem;
        letter-spacing: 0.13em;
        backdrop-filter: blur(10px);
    }

    .home-kicker::before {
        content: "";
        width: 0.45rem;
        height: 0.45rem;
        border-radius: 50%;
        background: #75d8c8;
        box-shadow: 0 0 0 4px rgba(117, 216, 200, 0.12);
    }

    .home-copy h1 {
        max-width: 760px;
        margin: 0 0 1rem;
        color: #ffffff !important;
        font-size: clamp(2.55rem, 4.8vw, 5rem) !important;
        line-height: 0.98 !important;
        letter-spacing: -0.055em !important;
    }

    .home-copy h1::before {
        display: none;
    }

    .home-copy p {
        max-width: 760px;
        color: #cbd4e3 !important;
        font-size: clamp(1rem, 1.5vw, 1.16rem);
        line-height: 1.65;
    }

    .home-stat-row {
        gap: 0.7rem;
        margin-top: 1.5rem;
    }

    .home-stat {
        min-height: 6.2rem;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.055);
        backdrop-filter: blur(10px);
    }

    .home-stat strong {
        color: #ffffff;
        font-size: 0.98rem;
    }

    .home-stat span {
        color: #9facc0;
        font-size: 0.78rem;
        line-height: 1.4;
    }

    .home-action-title {
        margin: 2rem 0 0.7rem;
        color: var(--aura-ink);
        font-size: 0.76rem;
        font-weight: 820;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    [data-testid="stMain"]:has(.home-hero) div.st-key-home_login button {
        min-height: 3.15rem;
        border-color: var(--aura-indigo) !important;
        background: linear-gradient(135deg, #7167ff, #5950e6) !important;
        color: #ffffff !important;
        box-shadow: 0 12px 28px rgba(102, 92, 246, 0.23);
    }

    [data-testid="stMain"]:has(.home-hero) div.st-key-home_register button {
        min-height: 3.15rem;
        border-color: var(--aura-line-strong) !important;
        background: rgba(255, 255, 255, 0.82) !important;
        color: var(--aura-ink) !important;
    }

    .home-feature-grid {
        gap: 0.9rem;
        margin-top: 1rem;
    }

    .home-feature {
        position: relative;
        overflow: hidden;
        min-height: 10.5rem;
        padding: 1.35rem 1.4rem;
        border: 1px solid rgba(23, 32, 51, 0.09);
        border-radius: 20px;
        background: rgba(255, 255, 255, 0.82);
        box-shadow: var(--aura-shadow-sm);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .home-feature:hover {
        transform: translateY(-3px);
        box-shadow: var(--aura-shadow-md);
    }

    .home-feature::before {
        content: attr(data-index);
        display: flex;
        align-items: center;
        justify-content: center;
        width: 2rem;
        height: 2rem;
        margin-bottom: 1.1rem;
        border-radius: 10px;
        background: var(--aura-indigo-soft);
        color: var(--aura-indigo-dark);
        font-size: 0.72rem;
        font-weight: 840;
    }

    .home-feature h3 {
        color: var(--aura-ink);
        font-size: 1.02rem !important;
    }

    .home-feature p {
        color: var(--aura-muted);
        font-size: 0.88rem;
        line-height: 1.55;
    }

    [data-testid="stMain"]:has(.auth-page-marker) {
        background:
            radial-gradient(circle at 50% -8%, rgba(102, 92, 246, 0.17), transparent 28rem),
            var(--aura-canvas);
    }

    [data-testid="stMain"]:has(.auth-page-marker) [data-testid="stMainBlockContainer"],
    [data-testid="stMain"]:has(.auth-page-marker) .block-container {
        max-width: 680px;
        margin-top: clamp(1rem, 5vh, 4rem);
        margin-bottom: 4rem;
        padding: clamp(1.6rem, 5vw, 3.2rem);
        border: 1px solid rgba(23, 32, 51, 0.09);
        border-radius: 26px;
        background: rgba(255, 255, 255, 0.92);
        box-shadow: 0 30px 85px rgba(23, 32, 51, 0.13);
        backdrop-filter: blur(18px);
    }

    [data-testid="stMain"]:has(.auth-page-marker) h1::before {
        content: "AURA  /  SECURE ACCESS";
    }

    [data-testid="stMain"]:has(.auth-page-marker) [data-testid="stImage"] {
        display: flex;
        justify-content: center;
        margin-bottom: 0.7rem;
    }

    [data-testid="stMain"]:has(.auth-page-marker) [data-testid="stImage"] img {
        padding: 0.55rem;
        border: 1px solid var(--aura-line);
        border-radius: 20px;
        background: #ffffff;
        box-shadow: var(--aura-shadow-sm);
    }

    .detail-hero {
        border: 1px solid rgba(255, 255, 255, 0.10);
        border-radius: 20px;
        background:
            radial-gradient(circle at 92% 0%, rgba(102, 92, 246, 0.42), transparent 18rem),
            linear-gradient(145deg, #172943, #101b2d);
        box-shadow: var(--aura-shadow-md);
    }

    .detail-hero h2,
    .detail-hero .detail-value {
        color: #ffffff !important;
    }

    .detail-eyebrow {
        color: #9f97ff;
    }

    .detail-subtitle {
        color: #c5cfde;
    }

    .detail-item,
    .detail-list-card {
        border-color: rgba(23, 32, 51, 0.09);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.90);
        box-shadow: 0 7px 22px rgba(23, 32, 51, 0.05);
    }

    .detail-label {
        color: var(--aura-muted);
    }

    .detail-value,
    .student-header-cell,
    .student-row-primary,
    .student-probability,
    .pager-label {
        color: var(--aura-ink);
    }

    .student-row-subtle,
    .student-list-meta {
        color: var(--aura-muted);
    }

    .student-row-divider {
        border-color: var(--aura-line);
    }

    /* High-contrast semantic states for the light interface. */
    .questionnaire-badge,
    .notification-badge {
        font-weight: 800;
        line-height: 1.2;
        text-shadow: none;
    }

    .questionnaire-completed {
        border-color: #86c99a !important;
        background: #dcf5e4 !important;
        color: #14532d !important;
    }

    .questionnaire-missing {
        border-color: #e8b66b !important;
        background: #fff1d6 !important;
        color: #854d0e !important;
    }

    .notification-not-required {
        border-color: #aeb6c4 !important;
        background: #edf0f4 !important;
        color: #374151 !important;
    }

    .notification-pending {
        border-color: #e8b66b !important;
        background: #fff1d6 !important;
        color: #854d0e !important;
    }

    .notification-sent,
    .notification-resent {
        border-color: #86c99a !important;
        background: #dcf5e4 !important;
        color: #14532d !important;
    }

    .notification-failed {
        border-color: #e79a9a !important;
        background: #fde2e2 !important;
        color: #8b1e1e !important;
    }

    .risk-summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin: 0.5rem 0 1.25rem;
    }

    .risk-summary-card {
        position: relative;
        overflow: hidden;
        min-height: 8.7rem;
        padding: 1.2rem 1.3rem;
        border: 1px solid var(--risk-border);
        border-radius: 18px;
        background: var(--risk-bg);
        box-shadow: 0 10px 28px rgba(23, 32, 51, 0.07);
    }

    .risk-summary-card::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 0.38rem;
        background: var(--risk-accent);
    }

    .risk-summary-label {
        color: var(--risk-text);
        font-size: 0.76rem;
        font-weight: 820;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .risk-summary-value {
        margin-top: 0.65rem;
        color: var(--risk-text);
        font-size: clamp(2rem, 3vw, 2.8rem);
        font-weight: 820;
        line-height: 1;
        letter-spacing: -0.05em;
    }

    .risk-summary-caption {
        margin-top: 0.65rem;
        color: var(--risk-text);
        font-size: 0.78rem;
        font-weight: 650;
        opacity: 0.86;
    }

    .risk-summary-high {
        --risk-bg: #fde8e8;
        --risk-border: #e7a4a9;
        --risk-accent: #b4232f;
        --risk-text: #7f1d1d;
    }

    .risk-summary-medium {
        --risk-bg: #fff1d6;
        --risk-border: #e8b66b;
        --risk-accent: #b36b00;
        --risk-text: #754508;
    }

    .risk-summary-low {
        --risk-bg: #dcf5e4;
        --risk-border: #86c99a;
        --risk-accent: #208454;
        --risk-text: #14532d;
    }

    .prediction-overview {
        position: relative;
        overflow: hidden;
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(11rem, 15rem);
        gap: 1.5rem;
        align-items: center;
        margin: 1rem 0 1.1rem;
        padding: clamp(1.35rem, 3vw, 2rem);
        border: 1px solid var(--overview-border);
        border-radius: 22px;
        background:
            radial-gradient(circle at 92% 5%, var(--overview-glow), transparent 18rem),
            var(--overview-bg);
        box-shadow: 0 18px 48px rgba(23, 32, 51, 0.10);
    }

    .prediction-overview::before {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 0.48rem;
        background: var(--overview-accent);
    }

    .prediction-overview-eyebrow {
        color: var(--overview-text);
        font-size: 0.7rem;
        font-weight: 850;
        letter-spacing: 0.12em;
        text-transform: uppercase;
    }

    .prediction-overview-title {
        margin-top: 0.55rem;
        color: var(--overview-text);
        font-size: clamp(1.75rem, 3vw, 2.65rem);
        font-weight: 820;
        line-height: 1.04;
        letter-spacing: -0.05em;
    }

    .prediction-overview-copy {
        max-width: 700px;
        margin-top: 0.75rem;
        color: var(--overview-text);
        font-size: 0.95rem;
        font-weight: 620;
        line-height: 1.55;
        opacity: 0.86;
    }

    .prediction-overview-score {
        position: relative;
        z-index: 1;
        padding: 1.15rem 1.25rem;
        border: 1px solid var(--overview-border);
        border-radius: 18px;
        background: rgba(255, 255, 255, 0.68);
        text-align: center;
        box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.75);
        backdrop-filter: blur(10px);
    }

    .prediction-overview-score strong {
        display: block;
        color: var(--overview-text);
        font-size: clamp(2.2rem, 4vw, 3.4rem);
        font-weight: 840;
        line-height: 1;
        letter-spacing: -0.06em;
    }

    .prediction-overview-score span {
        display: block;
        margin-top: 0.55rem;
        color: var(--overview-text);
        font-size: 0.69rem;
        font-weight: 840;
        letter-spacing: 0.09em;
        text-transform: uppercase;
    }

    .prediction-overview-critical,
    .prediction-overview-high {
        --overview-bg: #fde8e8;
        --overview-border: #e7a4a9;
        --overview-accent: #b4232f;
        --overview-text: #7f1d1d;
        --overview-glow: rgba(180, 35, 47, 0.14);
    }

    .prediction-overview-medium {
        --overview-bg: #fff1d6;
        --overview-border: #e8b66b;
        --overview-accent: #b36b00;
        --overview-text: #754508;
        --overview-glow: rgba(179, 107, 0, 0.14);
    }

    .prediction-overview-low {
        --overview-bg: #dcf5e4;
        --overview-border: #86c99a;
        --overview-accent: #208454;
        --overview-text: #14532d;
        --overview-glow: rgba(32, 132, 84, 0.14);
    }

    .prediction-facts {
        display: grid;
        grid-template-columns: repeat(5, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1.1rem;
    }

    .prediction-fact {
        min-height: 5.7rem;
        padding: 0.95rem 1rem;
        border: 1px solid rgba(23, 32, 51, 0.10);
        border-radius: 15px;
        background: rgba(255, 255, 255, 0.90);
        box-shadow: 0 7px 22px rgba(23, 32, 51, 0.05);
    }

    .prediction-fact-label {
        color: #596377;
        font-size: 0.67rem;
        font-weight: 830;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }

    .prediction-fact-value {
        margin-top: 0.5rem;
        color: var(--aura-ink);
        font-size: 0.98rem;
        font-weight: 770;
        line-height: 1.3;
        overflow-wrap: anywhere;
    }

    .risk-card,
    .risk-card h1,
    .risk-card h2,
    .risk-card h3,
    .risk-card h4,
    .risk-card p,
    .risk-card span {
        color: #ffffff !important;
    }

    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] *,
    .student-row-subtle,
    .student-list-meta,
    .detail-label {
        color: #525c6f !important;
        -webkit-text-fill-color: #525c6f !important;
        opacity: 1 !important;
    }

    [data-testid="stAlert"] *,
    [data-testid="stNotification"] * {
        color: #263247 !important;
        -webkit-text-fill-color: #263247 !important;
        opacity: 1 !important;
    }

    a {
        color: #075eae;
        text-decoration-thickness: 1px;
        text-underline-offset: 2px;
    }

    div[data-testid="stButton"] > button:disabled,
    div[data-testid="stDownloadButton"] > button:disabled,
    button:disabled {
        border-color: #b9bfca !important;
        background: #e9ebef !important;
        color: #4f596b !important;
        -webkit-text-fill-color: #4f596b !important;
        opacity: 1 !important;
        box-shadow: none !important;
    }

    div[data-testid="stButton"] > button:disabled *,
    div[data-testid="stDownloadButton"] > button:disabled *,
    button:disabled * {
        color: #4f596b !important;
        -webkit-text-fill-color: #4f596b !important;
        opacity: 1 !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: #687286 !important;
        -webkit-text-fill-color: #687286 !important;
        opacity: 1 !important;
    }

    .stChatMessage,
    [data-testid="stChatMessage"] {
        border: 1px solid rgba(23, 32, 51, 0.09) !important;
        border-radius: 18px !important;
        background: rgba(255, 255, 255, 0.92) !important;
        box-shadow: var(--aura-shadow-sm);
    }

    .stChatMessage p,
    .stChatMessage li,
    .stChatMessage span,
    .stChatMessage div {
        color: var(--aura-ink) !important;
    }

    div[class*="st-key-quick_"] button {
        border-color: #c9c5ff !important;
        background: var(--aura-indigo-soft) !important;
        color: var(--aura-indigo-dark) !important;
        box-shadow: none !important;
    }

    div[class*="st-key-quick_"] button p {
        color: inherit !important;
    }

    div[class*="st-key-fab_chat"] button {
        border-color: var(--aura-indigo) !important;
        background: linear-gradient(135deg, #7167ff, #5950e6) !important;
        color: #ffffff !important;
        box-shadow: 0 13px 30px rgba(102, 92, 246, 0.25) !important;
    }

    .aura-back-top {
        border-color: rgba(102, 92, 246, 0.38);
        background: rgba(255, 255, 255, 0.92);
        color: var(--aura-indigo-dark) !important;
        box-shadow: 0 12px 28px rgba(23, 32, 51, 0.13);
        backdrop-filter: blur(12px);
    }

    /*
       Hard light-mode readability layer.
       Streamlit can inherit dark widget canvases from the active theme/browser,
       so these rules keep every input, graph, select menu, and table readable.
    */
    .stApp *,
    [data-testid="stAppViewContainer"] * {
        caret-color: var(--aura-indigo);
    }

    [data-testid="stTextInput"],
    [data-testid="stNumberInput"],
    [data-testid="stTextArea"],
    [data-testid="stSelectbox"],
    [data-testid="stMultiSelect"],
    [data-testid="stDateInput"],
    [data-testid="stTimeInput"],
    [data-testid="stFileUploader"],
    [data-testid="stSlider"],
    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    [data-testid="stPlotlyChart"] {
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: initial !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input,
    [data-testid="stDateInput"] input,
    [data-testid="stTimeInput"] input,
    [data-testid="stTextArea"] textarea,
    [data-testid="stSelectbox"] [role="combobox"],
    [data-testid="stMultiSelect"] [role="combobox"],
    [data-testid="stSelectbox"] input,
    [data-testid="stMultiSelect"] input,
    [data-baseweb="select"] span,
    [data-baseweb="select"] div,
    [data-baseweb="input"] input,
    [data-baseweb="textarea"] textarea {
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
        opacity: 1 !important;
    }

    [data-baseweb="input"],
    [data-baseweb="input"] > div,
    [data-baseweb="textarea"],
    [data-baseweb="textarea"] > div,
    [data-baseweb="select"],
    [data-baseweb="select"] > div,
    [data-testid="stNumberInput"] [data-testid="stNumberInputContainer"] {
        background: #ffffff !important;
        color: var(--aura-ink) !important;
        border-color: #aeb6c4 !important;
        box-shadow: 0 2px 8px rgba(23, 32, 51, 0.045) !important;
    }

    [data-testid="stNumberInput"] button,
    [data-testid="stNumberInput"] button:hover,
    [data-testid="stNumberInput"] button:focus,
    [data-testid="stNumberInput"] button:active {
        background: #f3f5fa !important;
        border-color: #aeb6c4 !important;
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
    }

    [data-testid="stNumberInput"] button svg,
    [data-testid="stSelectbox"] svg,
    [data-testid="stMultiSelect"] svg,
    [data-testid="stDateInput"] svg,
    [data-testid="stTimeInput"] svg {
        color: #374151 !important;
        fill: #374151 !important;
        opacity: 1 !important;
    }

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"],
    ul[role="listbox"] {
        background: #ffffff !important;
        color: var(--aura-ink) !important;
        border: 1px solid #cfd1d8 !important;
        box-shadow: 0 22px 60px rgba(23, 32, 51, 0.16) !important;
    }

    [role="option"],
    [role="option"] *,
    [data-baseweb="menu"] li,
    [data-baseweb="menu"] li * {
        background: #ffffff !important;
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
        opacity: 1 !important;
    }

    [role="option"]:hover,
    [role="option"][aria-selected="true"],
    [data-baseweb="menu"] li:hover {
        background: #eeecff !important;
        color: #332bb8 !important;
        -webkit-text-fill-color: #332bb8 !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"],
    [data-testid="stSlider"] > div {
        color: var(--aura-ink) !important;
    }

    [data-testid="stSlider"] [data-testid="stTickBar"] *,
    [data-testid="stSlider"] [data-testid="stThumbValue"],
    [data-testid="stSlider"] [role="slider"] {
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
    }

    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderDropzone"] * {
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
        opacity: 1 !important;
    }

    [data-testid="stPlotlyChart"] {
        background:
            linear-gradient(145deg, rgba(255,255,255,0.98), rgba(249,250,253,0.98)) !important;
    }

    [data-testid="stPlotlyChart"] .js-plotly-plot,
    [data-testid="stPlotlyChart"] .plot-container,
    [data-testid="stPlotlyChart"] .svg-container {
        background: transparent !important;
    }

    [data-testid="stPlotlyChart"] text {
        fill: var(--aura-ink) !important;
        color: var(--aura-ink) !important;
    }

    [data-testid="stDataFrame"],
    [data-testid="stTable"],
    div[data-testid="stDataFrame"] > div,
    div[data-testid="stTable"] > div {
        background: #ffffff !important;
        color: var(--aura-ink) !important;
    }

    [data-testid="stDataFrame"] *,
    [data-testid="stTable"] *,
    .stDataFrame *,
    .stTable * {
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
        border-color: #d8dbe4 !important;
        opacity: 1 !important;
    }

    [data-testid="stDataFrame"] canvas,
    [data-testid="stTable"] canvas {
        background: #ffffff !important;
    }

    [data-testid="stDataFrame"] thead,
    [data-testid="stTable"] thead,
    [data-testid="stDataFrame"] [role="columnheader"],
    [data-testid="stTable"] [role="columnheader"] {
        background: #edf0ff !important;
        color: #172033 !important;
        -webkit-text-fill-color: #172033 !important;
        font-weight: 800 !important;
    }

    [data-testid="stDataFrame"] tbody tr:nth-child(even),
    [data-testid="stTable"] tbody tr:nth-child(even) {
        background: #fafbff !important;
    }

    [data-testid="stDataFrame"] tbody tr:hover,
    [data-testid="stTable"] tbody tr:hover {
        background: #f0efff !important;
    }

    [data-testid="stTable"] table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: #ffffff !important;
    }

    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        padding: 0.65rem 0.8rem !important;
        border-bottom: 1px solid #e3e5ec !important;
        color: var(--aura-ink) !important;
        -webkit-text-fill-color: var(--aura-ink) !important;
    }

    @media (max-width: 900px) {
        section[data-testid="stSidebar"] {
            width: min(20.5rem, 88vw) !important;
            min-width: min(20.5rem, 88vw) !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] {
            transform: translateX(-100%) !important;
            pointer-events: none !important;
        }

        section[data-testid="stSidebar"][aria-expanded="true"] {
            transform: translateX(0) !important;
            pointer-events: auto !important;
        }

        section[data-testid="stSidebar"][aria-expanded="true"]
        button[data-testid="stBaseButton-headerNoPadding"] {
            display: flex !important;
        }

        [data-testid="stMainBlockContainer"],
        .block-container {
            padding: 2rem 1.1rem 4rem;
        }

        .home-hero {
            grid-template-columns: 1fr;
            border-radius: 23px;
        }

        .home-logo-card {
            width: min(100%, 240px);
            min-height: 190px;
            margin: 0 auto;
        }

        .home-copy h1 {
            font-size: clamp(2.5rem, 12vw, 4rem) !important;
        }

        .home-stat-row,
        .home-feature-grid,
        .risk-summary-grid,
        .prediction-facts {
            grid-template-columns: 1fr;
        }

        .prediction-overview {
            grid-template-columns: 1fr;
        }

        .prediction-overview-score {
            width: 100%;
        }

        [data-testid="stMetric"] {
            min-height: auto;
        }
    }

    @media (max-width: 600px) {
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 0.75rem !important;
        }

        [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
            flex: 1 1 100% !important;
            width: 100% !important;
            min-width: 100% !important;
        }

        [data-testid="stMainBlockContainer"],
        .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }

        [data-testid="stTabs"] [role="tablist"] {
            justify-content: flex-start !important;
            overflow-x: auto !important;
            overflow-y: hidden !important;
            scrollbar-width: thin;
        }

        [data-testid="stTabs"] button[data-baseweb="tab"],
        [data-testid="stTabs"] button[data-baseweb="tab"] * {
            flex: 0 0 auto !important;
            min-width: max-content !important;
            white-space: nowrap !important;
            overflow-wrap: normal !important;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *,
        *::before,
        *::after {
            scroll-behavior: auto !important;
            transition-duration: 0.01ms !important;
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def init_session():
    defaults = {
        "logged_in": False,
        "user_email": "",
        "user_role": "",
        "user_staff_id": "",
        "page": "home",
        "pending_auth_email": "",
        "pending_auth_role": "",
        "pending_auth_staff_id": "",
        "pending_reset_email": "",
        "pending_reset_staff_id": "",
        "reset_otp": "",
        "reset_otp_expiry": None,
        "reset_email_sent": False,
        "reset_error_message": "",
        "login_success_message": "",
        "delete_account_candidate": None,
        "manage_users_message": "",
        "otp": "",
        "otp_expiry": None,
        "otp_email_sent": False,
        "otp_error_message": "",
        "otp_last_sent_at": None,
        "otp_last_sent_email": "",
        "otp_send_in_progress": False,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def show_logo(width=140):
    if LOGO_FILE.exists():
        st.image(str(LOGO_FILE), width=width)


@st.cache_data(show_spinner=False)
def image_data_uri(path_text):
    path = Path(path_text)
    if not path.exists():
        return ""
    suffix = path.suffix.lower().lstrip(".") or "jpeg"
    if suffix == "jpg":
        suffix = "jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/{suffix};base64,{encoded}"


def show_sidebar_logo():
    logo_uri = image_data_uri(str(LOGO_FILE))
    if not logo_uri:
        return
    st.markdown(
        f"""
        <div class="sidebar-logo-wrap">
            <img src="{logo_uri}" alt="AURA logo">
            <div>
                <div class="sidebar-brand-name">AURA</div>
                <div class="sidebar-brand-subtitle">Student success intelligence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def scroll_to_top_if_route_changed(route_key):
    if st.session_state.get("_last_visible_route_key") == route_key:
        return

    st.session_state["_last_visible_route_key"] = route_key
    st.iframe(
        """
        <script>
        function auraScrollToTop() {
            try {
                const parentWindow = window.parent;
                const parentDocument = parentWindow.document;
                const targets = [
                    parentWindow,
                    parentDocument.documentElement,
                    parentDocument.body,
                    parentDocument.querySelector('[data-testid="stAppViewContainer"]'),
                    parentDocument.querySelector('[data-testid="stMain"]'),
                    parentDocument.querySelector('section[data-testid="stMain"]'),
                    parentDocument.querySelector('section.main'),
                    parentDocument.querySelector('main')
                ];
                targets.forEach((target) => {
                    if (target && typeof target.scrollTo === 'function') {
                        target.scrollTo({ top: 0, left: 0, behavior: 'auto' });
                    }
                    if (target && 'scrollTop' in target) {
                        target.scrollTop = 0;
                    }
                });
                parentDocument.querySelectorAll('*').forEach((element) => {
                    const hasScroll = element.scrollHeight > element.clientHeight;
                    if (hasScroll && element.scrollTop > 0) {
                        element.scrollTop = 0;
                    }
                });
            } catch (error) {}
        }
        auraScrollToTop();
        requestAnimationFrame(auraScrollToTop);
        setTimeout(auraScrollToTop, 100);
        setTimeout(auraScrollToTop, 300);
        setTimeout(auraScrollToTop, 800);
        </script>
        """,
        height=1,
        width=1,
    )


def render_page_tools(route_key, show_back_to_top=True):
    scroll_to_top_if_route_changed(route_key)
    back_to_top_link = ""
    if show_back_to_top:
        scroll_click = (
            "event.preventDefault();"
            "(function(){"
            "const d=document;"
            "const targets=[window,d.documentElement,d.body,"
            "d.querySelector('[data-testid=stAppViewContainer]'),"
            "d.querySelector('[data-testid=stMain]'),"
            "d.querySelector('section[data-testid=stMain]'),"
            "d.querySelector('section.main'),d.querySelector('main')];"
            "targets.forEach(function(t){"
            "if(t&&typeof t.scrollTo==='function'){t.scrollTo({top:0,left:0,behavior:'smooth'});}"
            "if(t&&'scrollTop' in t){t.scrollTop=0;}"
            "});"
            "d.querySelectorAll('*').forEach(function(e){"
            "if(e.scrollHeight>e.clientHeight){"
            "e.scrollTop=0;"
            "if(typeof e.scrollTo==='function'){e.scrollTo({top:0,left:0,behavior:'smooth'});}"
            "}"
            "});"
            "})();"
        )
        back_to_top_link = (
            '<a class="aura-back-top" href="#aura-page-top" '
            'aria-label="Back to top" title="Back to top" '
            f'onclick="{scroll_click}">&#8593;</a>'
        )
    st.markdown(
        f'<div id="aura-page-top"></div>{back_to_top_link}',
        unsafe_allow_html=True,
    )


def is_valid_email(email):
    return re.match(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$", email) is not None


def normalize_staff_id(staff_id):
    return _normalize_staff_id_text(staff_id)


def staff_id_match_key(staff_id):
    normalized = normalize_staff_id(staff_id)
    return normalized.lstrip("0") or normalized


def role_display_name(role):
    if role in {"admin", "administrator"}:
        return "Administrator"
    if role == "advisor":
        return "Advisor"
    if role == "student":
        return "Student"
    return str(role or "").title()


def normalize_account_role(role):
    role = str(role or "").strip().lower()
    if role == "admin":
        return "administrator"
    return role


def validate_institutional_account(email, staff_id, role):
    email = str(email or "").strip().lower()
    staff_id = normalize_staff_id(staff_id).upper()
    role = normalize_account_role(role)
    domain = ROLE_EMAIL_DOMAINS.get(role)

    if not is_valid_email(email):
        return False, "Incorrect register email: enter a valid email address."

    if not domain:
        return False, "Please choose Administrator or Advisor."

    if role in {"administrator", "admin"}:
        if not re.fullmatch(r"ADM\d{3}", staff_id):
            return False, "Incorrect ID number: use format ADM001."

    elif role == "advisor":
        if not re.fullmatch(r"ADV\d{3}", staff_id):
            return False, "Incorrect ID number: use format ADV001."

    expected_suffix = f"@{domain}"
    if not email.endswith(expected_suffix):
        return False, "Incorrect register email: this email domain is not allowed."

    email_prefix = email.split("@")[0].upper()
    if email_prefix != staff_id:
        return False, "Email and ID must match. Example: ADM003 with ADM003@aura-student-risk.com."

    return True, ""


def read_authorized_users_csv():
    if not AUTHORIZED_USERS_FILE.exists():
        return False, "authorized_users.csv was not found.", pd.DataFrame()

    try:
        authorized_df = pd.read_csv(
            AUTHORIZED_USERS_FILE,
            dtype={"staff_id": str, "email": str, "role": str, "is_active": str},
            keep_default_na=False,
        ).fillna("")
    except Exception as error:
        return False, f"Could not read authorized_users.csv: {error}", pd.DataFrame()

    required_columns = {"staff_id", "email", "role", "is_active"}
    missing_columns = sorted(required_columns - set(authorized_df.columns))
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        return False, f"authorized_users.csv is missing required column(s): {missing_text}.", pd.DataFrame()

    authorized_df = authorized_df.copy()
    authorized_df["staff_id"] = authorized_df["staff_id"].apply(lambda x: normalize_staff_id(x).upper())
    authorized_df["email"] = authorized_df["email"].astype(str).str.strip().str.lower()
    authorized_df["role"] = authorized_df["role"].astype(str).str.strip().str.lower()
    authorized_df["is_active"] = authorized_df["is_active"].apply(
        lambda value: 0 if str(value).strip() == "0" else 1
    )

    authorized_df = authorized_df[
        (authorized_df["staff_id"] != "") &
        (authorized_df["email"] != "") &
        (authorized_df["role"] != "")
    ]

    return True, "", authorized_df


def refresh_authorized_users_from_csv():
    ok, message, authorized_df = read_authorized_users_csv()
    if not ok:
        return False, message

    with get_conn() as conn:
        conn.execute("DELETE FROM authorized_users")

        for _, row in authorized_df.iterrows():
            staff_id = normalize_staff_id(row["staff_id"]).upper()
            email = str(row["email"]).strip().lower()
            role = normalize_account_role(row["role"])
            is_active = int(row["is_active"])
            name = staff_id

            if not staff_id or not email or not role:
                continue

            conn.execute(
                """
                INSERT OR REPLACE INTO authorized_users
                    (name, email, staff_id, role, is_active)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, email, staff_id, role, is_active),
            )

        conn.commit()

    return True, ""


def explain_authorized_registration_error(email, staff_id):
    ok, message, authorized_df = read_authorized_users_csv()
    if not ok:
        return message

    if authorized_df.empty:
        return "No approved users are listed in authorized_users.csv."

    email = str(email or "").strip().lower()
    staff_id_key = staff_id_match_key(staff_id)
    authorized_df = authorized_df.copy()
    authorized_df["_staff_id_key"] = authorized_df["staff_id"].apply(staff_id_match_key)
    email_matches = authorized_df[authorized_df["email"] == email]
    id_matches = authorized_df[authorized_df["_staff_id_key"] == staff_id_key]
    exact_matches = authorized_df[
        (authorized_df["email"] == email)
        & (authorized_df["_staff_id_key"] == staff_id_key)
    ]

    if exact_matches.empty:
        if email_matches.empty and id_matches.empty:
            return "This email and ID number are not approved in authorized_users.csv."
        if email_matches.empty:
            return "This email is not approved in authorized_users.csv."
        if id_matches.empty:
            return "This ID number is not approved in authorized_users.csv."
        return "The email and ID number do not match the same approved CSV record."

    if not exact_matches["is_active"].astype(int).eq(1).any():
        return "This approved CSV record is inactive."

    return "This approved CSV record could not be loaded into the database."


def lookup_authorized_user(email, staff_id, role=None):
    email = str(email or "").strip().lower()
    staff_id_key = staff_id_match_key(staff_id)
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT name, email, staff_id, role, is_active
            FROM authorized_users
            WHERE lower(email) = lower(?)
              AND is_active = 1
            """,
            (email,),
        ).fetchall()
    for row in rows:
        if staff_id_match_key(row[2]) == staff_id_key:
            return row
    return None


def load_authorized_users():
    with get_conn() as conn:
        return pd.read_sql_query(
            """
            SELECT name, email, staff_id, role, is_active, created_at
            FROM authorized_users
            ORDER BY role, staff_id
            """,
            conn,
        )


def sync_authorized_users_csv():
    authorized_df = load_authorized_users()
    columns = ["staff_id", "email", "role", "is_active"]
    if authorized_df.empty:
        pd.DataFrame(columns=columns).to_csv(AUTHORIZED_USERS_FILE, index=False)
    else:
        authorized_df[columns].to_csv(AUTHORIZED_USERS_FILE, index=False)


def institutional_role_prefix(role):
    role = normalize_account_role(role)
    if role in {"administrator", "admin"}:
        return "ADM"
    if role == "advisor":
        return "ADV"
    return ""


def institutional_email_for_staff_id(staff_id):
    return f"{str(staff_id or '').strip().upper()}@aura-student-risk.com".lower()


def used_staff_ids_and_emails(conn, exclude_email=""):
    exclude_email = str(exclude_email or "").strip().lower()
    used_ids = set()
    used_emails = set()
    for table in ("users", "authorized_users"):
        for row in conn.execute(f"SELECT staff_id, email FROM {table}"):
            staff_id = normalize_staff_id(row[0]).upper()
            email = str(row[1] or "").strip().lower()
            if exclude_email and email == exclude_email:
                continue
            if staff_id:
                used_ids.add(staff_id)
            if email:
                used_emails.add(email)
    return used_ids, used_emails


def allocate_institutional_identity(conn, current_staff_id, target_role, exclude_email=""):
    prefix = institutional_role_prefix(target_role)
    if not prefix:
        return None, None

    used_ids, used_emails = used_staff_ids_and_emails(conn, exclude_email)
    current_staff_id = normalize_staff_id(current_staff_id).upper()
    digit_match = re.search(r"(\d+)$", current_staff_id)
    preferred_number = int(digit_match.group(1)) if digit_match else 1
    if preferred_number < 1 or preferred_number > 999:
        preferred_number = random.randint(1, 999)

    candidate_numbers = [preferred_number] + random.sample(
        [number for number in range(1, 1000) if number != preferred_number],
        998,
    )
    for number in candidate_numbers:
        staff_id = f"{prefix}{number:03d}"
        email = institutional_email_for_staff_id(staff_id)
        if staff_id not in used_ids and email not in used_emails:
            return staff_id, email
    return None, None


def infer_staff_role_from_id(staff_id):
    staff_id = normalize_staff_id(staff_id).upper()
    if re.fullmatch(r"ADM\d{3}", staff_id):
        return "administrator"
    if re.fullmatch(r"ADV\d{3}", staff_id):
        return "advisor"
    return ""


def add_authorized_user(email, staff_id):
    staff_id = normalize_staff_id(staff_id)
    role = infer_staff_role_from_id(staff_id)
    if not role:
        return False
    with get_conn() as conn:
        try:
            conn.execute(
                """
                INSERT INTO authorized_users (
                    name, email, staff_id, role, is_active, created_at
                )
                VALUES (?, ?, ?, ?, 1, ?)
                ON CONFLICT(email) DO UPDATE SET
                    name = excluded.name,
                    staff_id = excluded.staff_id,
                    is_active = 1
                """,
                (
                    staff_id,
                    str(email or "").strip().lower(),
                    staff_id,
                    role,
                    app_now_text(),
                ),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            return False
    sync_authorized_users_csv()
    return True


def update_authorized_user_role(email, new_role):
    email = str(email or "").strip().lower()
    new_role = normalize_account_role(new_role)
    with get_conn() as conn:
        auth_row = conn.execute(
            "SELECT name, staff_id, role FROM authorized_users WHERE lower(email) = lower(?)",
            (email,),
        ).fetchone()
        user_row = conn.execute(
            "SELECT name, staff_id, role FROM users WHERE lower(email) = lower(?)",
            (email,),
        ).fetchone()
        if auth_row is None and user_row is None:
            return False, "User not found."

        current_staff_id = auth_row[1] if auth_row is not None else user_row[1]
        new_staff_id, new_email = allocate_institutional_identity(
            conn, current_staff_id, new_role, exclude_email=email
        )
        if not new_staff_id or not new_email:
            return False, "No available institutional ID could be generated."

        if auth_row is not None:
            conn.execute(
                """
                UPDATE authorized_users
                SET email = ?, staff_id = ?, role = ?, name = ?
                WHERE lower(email) = lower(?)
                """,
                (new_email, new_staff_id, new_role, new_staff_id, email),
            )
        if user_row is not None:
            conn.execute(
                """
                UPDATE users
                SET email = ?, staff_id = ?, role = ?, name = ?
                WHERE lower(email) = lower(?)
                """,
                (new_email, new_staff_id, new_role, new_staff_id, email),
            )
        conn.commit()
    sync_authorized_users_csv()
    return True, f"Role updated. New ID: {new_staff_id}. New Email: {new_email}."


def delete_registered_account(email):
    email = str(email or "").strip().lower()
    if not email:
        return False, "No account email was selected."
    if email == str(st.session_state.get("user_email", "")).strip().lower():
        return False, "You cannot remove the account you are currently signed in with."

    with get_conn() as conn:
        user_row = conn.execute(
            "SELECT email FROM users WHERE lower(email) = lower(?)",
            (email,),
        ).fetchone()
        if user_row is None:
            return False, "Registered account not found."

        conn.execute("DELETE FROM users WHERE lower(email) = lower(?)", (email,))
        conn.commit()

    return True, "Account removed."


def lookup_user_for_reset(email, staff_id):
    email = str(email or "").strip().lower()
    staff_id = normalize_staff_id(staff_id)
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT role
            FROM users
            WHERE lower(email) = lower(?) AND staff_id = ?
            """,
            (email, staff_id),
        ).fetchone()


def lookup_user_for_reset_by_email(email):
    email = str(email or "").strip().lower()
    with get_conn() as conn:
        return conn.execute(
            """
            SELECT role, staff_id
            FROM users
            WHERE lower(email) = lower(?)
            """,
            (email,),
        ).fetchone()


def update_user_password(email, staff_id, new_password):
    with get_conn() as conn:
        cursor = conn.execute(
            """
            UPDATE users
            SET password = ?, password_hashed = 1
            WHERE lower(email) = lower(?) AND staff_id = ?
            """,
            (hash_password(new_password), str(email or "").strip().lower(), normalize_staff_id(staff_id)),
        )
        return cursor.rowcount > 0


def hash_password(password):
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password, saved_password, password_hashed):
    if int(password_hashed or 0) == 1:
        try:
            return bcrypt.checkpw(
                password.encode("utf-8"), saved_password.encode("utf-8")
            )
        except ValueError:
            return False
    return saved_password == password


def normalize_sender_email():
    sender = EMAIL_FROM.strip()
    if "<" in sender and ">" in sender:
        return sender
    return f"{EMAIL_SENDER_NAME} <{sender}>"


def build_otp_email_text(otp):
    return f"""
Hello,

Your AURA verification code is: {otp}

This code will expire in {OTP_EXPIRY_MINUTES} minutes.

AURA Student At-Risk Prediction System
"""


def build_otp_email_html(otp):
    return f"""
    <div style="font-family: Arial, sans-serif; line-height: 1.6; color: #111827;">
        <h2>AURA OTP Verification</h2>
        <p>Hello,</p>
        <p>Your AURA verification code is:</p>
        <h1 style="letter-spacing: 4px; color: #1f77ff;">{otp}</h1>
        <p>This code will expire in {OTP_EXPIRY_MINUTES} minutes.</p>
        <p>AURA Student At-Risk Prediction System</p>
    </div>
    """


def send_otp_email(receiver_email, otp):
    if not RESEND_API_KEY:
        return False, "RESEND_API_KEY is missing in .streamlit/secrets.toml."

    payload = {
        "from": normalize_sender_email(),
        "to": [receiver_email.strip().lower()],
        "subject": "AURA OTP Verification Code",
        "text": build_otp_email_text(otp),
        "html": build_otp_email_html(otp),
    }

    request = urllib.request.Request(
        RESEND_API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {RESEND_API_KEY}",
            "Content-Type": "application/json",
            "User-Agent": "AURA-Streamlit/1.0",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
            return 200 <= response.status < 300, body
    except urllib.error.HTTPError as error:
        try:
            body = error.read().decode("utf-8")
        except Exception:
            body = str(error)
        return False, f"HTTPError {error.code}: {body}"
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def generate_and_send_otp(email, role, staff_id="", force_new=False):
    normalized_email = email.strip().lower()
    now = app_now()
    existing_expiry = st.session_state.get("otp_expiry")
    last_sent_at = st.session_state.get("otp_last_sent_at")
    has_valid_existing_otp = (
        not force_new
        and st.session_state.get("otp")
        and st.session_state.get("otp_last_sent_email") == normalized_email
        and existing_expiry is not None
        and now < existing_expiry
        and last_sent_at is not None
        and (now - last_sent_at).total_seconds() < 30
    )
    if has_valid_existing_otp:
        st.session_state.page = "otp"
        st.rerun()

    if st.session_state.get("otp_send_in_progress"):
        st.session_state.page = "otp"
        st.rerun()

    st.session_state.otp_send_in_progress = True
    otp = str(random.randint(100000, 999999))
    st.session_state.otp = otp
    st.session_state.otp_expiry = now + timedelta(
        minutes=OTP_EXPIRY_MINUTES
    )
    st.session_state.pending_auth_email = normalized_email
    st.session_state.pending_auth_role = role
    st.session_state.pending_auth_staff_id = normalize_staff_id(staff_id)

    try:
        sent, message = send_otp_email(normalized_email, otp)
        st.session_state.otp_email_sent = sent
        st.session_state.otp_error_message = message
        if sent or ALLOW_LOCAL_OTP_FALLBACK:
            st.session_state.otp_last_sent_at = now
            st.session_state.otp_last_sent_email = normalized_email
    finally:
        st.session_state.otp_send_in_progress = False

    if sent or ALLOW_LOCAL_OTP_FALLBACK:
        st.session_state.page = "otp"
        st.rerun()

    st.error("OTP could not be sent.")
    st.warning("Check RESEND_API_KEY and EMAIL_FROM in .streamlit/secrets.toml.")
    st.write("Reason:", message)


def generate_and_send_reset_otp(email, staff_id):
    otp = str(random.randint(100000, 999999))
    st.session_state.reset_otp = otp
    st.session_state.reset_otp_expiry = app_now() + timedelta(
        minutes=OTP_EXPIRY_MINUTES
    )
    st.session_state.pending_reset_email = email.strip().lower()
    st.session_state.pending_reset_staff_id = normalize_staff_id(staff_id)

    sent, message = send_otp_email(email, otp)
    st.session_state.reset_email_sent = sent
    st.session_state.reset_error_message = message

    if sent or ALLOW_LOCAL_OTP_FALLBACK:
        st.session_state.page = "reset_password"
        st.rerun()

    st.error("Password reset OTP could not be sent.")
    st.warning("Check RESEND_API_KEY and EMAIL_FROM in .streamlit/secrets.toml.")
    st.write("Reason:", message)


def create_user(name, email, password, role, staff_id):
    normalized_email = email.strip().lower()
    normalized_staff_id = normalize_staff_id(staff_id)
    hashed_password = hash_password(password)

    with get_conn() as conn:
        try:
            existing_row = conn.execute(
                """
                SELECT id, email, COALESCE(staff_id, '')
                FROM users
                WHERE lower(email) = lower(?)
                   OR staff_id = ?
                LIMIT 1
                """,
                (normalized_email, normalized_staff_id),
            ).fetchone()

            if existing_row is not None:
                _, existing_email, existing_staff_id = existing_row
                existing_email = str(existing_email or "").strip().lower()
                existing_staff_id = normalize_staff_id(existing_staff_id)
                if (
                    existing_email != normalized_email
                    or existing_staff_id != normalized_staff_id
                ):
                    if existing_email == normalized_email:
                        return "email_conflict"
                    if existing_staff_id == normalized_staff_id:
                        return "staff_id_conflict"
                    return "conflict"
                return "already_registered"

            conn.execute(
                """
                INSERT INTO users (
                    name, email, staff_id, password, role, password_hashed, created_at
                )
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    name.strip(),
                    normalized_email,
                    normalized_staff_id,
                    hashed_password,
                    role,
                    app_now_text(),
                ),
            )
            conn.commit()
            return "created"
        except sqlite3.IntegrityError:
            return "conflict"


def check_login(email, password):
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT password, role, password_hashed, staff_id
            FROM users
            WHERE lower(email) = lower(?)
            """,
            (email.strip().lower(),),
        ).fetchone()

    if row is None:
        return False, None, ""

    saved_password, role, password_hashed, staff_id = row
    if verify_password(password, saved_password, password_hashed):
        return True, role, normalize_staff_id(staff_id)
    return False, None, ""


def is_student_role(role):
    return str(role or "").strip().lower() == "student"


def normalize_student_id(student_id):
    return str(student_id or "").strip().upper()


def recommended_course_hours(credits):
    return float(credits) * 2


def course_study_status(credits, weekly_hours):
    required_hours = recommended_course_hours(credits)
    gap = max(0, required_hours - float(weekly_hours))
    status = "Needs more study time" if gap > 0 else "On track"
    return required_hours, gap, status


def save_student_questionnaire(student_email, student_id, answers, courses):
    normalized_email = str(student_email or "").strip().lower()
    normalized_student_id = normalize_student_id(student_id)
    questionnaire_semester = normalize_semester(answers.get("semester", "Semester 1"))

    with get_conn() as conn:
        cursor = conn.execute(
            """
            INSERT INTO student_questionnaire (
                student_email, student_id, semester, attendance_rate, assignment_status,
                coursework_mark, auto_tracking, performance_indicators,
                update_frequency, alert_types, suggestions_needed, alert_method,
                badge_motivation, badge_achievements, reward_encouragement,
                allow_risk_flagging, preferred_support, communication_method,
                communication_ease, main_barrier, administration_support,
                high_risk_reminder_consent, preferred_contact_method,
                preferred_contact_details, submitted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                normalized_email,
                normalized_student_id,
                questionnaire_semester,
                answers.get("attendance_rate", ""),
                answers.get("assignment_status", ""),
                answers.get("coursework_mark", ""),
                answers.get("auto_tracking", ""),
                json.dumps(answers.get("performance_indicators", [])),
                answers.get("update_frequency", ""),
                json.dumps(answers.get("alert_types", [])),
                json.dumps(answers.get("suggestions_needed", [])),
                json.dumps(answers.get("alert_method", [])),
                int(answers.get("badge_motivation", 3)),
                json.dumps(answers.get("badge_achievements", [])),
                int(answers.get("reward_encouragement", 3)),
                answers.get("allow_risk_flagging", ""),
                json.dumps(answers.get("preferred_support", [])),
                json.dumps(answers.get("communication_method", [])),
                int(answers.get("communication_ease", 3)),
                json.dumps(answers.get("main_barrier", [])),
                json.dumps(answers.get("administration_support", [])),
                answers.get("high_risk_reminder_consent", ""),
                json.dumps(answers.get("preferred_contact_method", [])),
                answers.get("preferred_contact_details", ""),
                app_now_text(),
            ),
        )
        questionnaire_id = cursor.lastrowid

        for course in courses:
            course_name = str(course.get("course_name", "")).strip()
            if not course_name:
                continue
            credits = float(course.get("course_credits", 0))
            weekly_hours = float(course.get("weekly_study_hours", 0))
            required_hours, gap, status = course_study_status(credits, weekly_hours)
            conn.execute(
                """
                INSERT INTO student_course_study_plan (
                    questionnaire_id, student_email, student_id, course_name,
                    course_credits, weekly_study_hours, recommended_weekly_hours,
                    study_gap, status, submitted_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    questionnaire_id,
                    normalized_email,
                    normalized_student_id,
                    course_name,
                    credits,
                    weekly_hours,
                    required_hours,
                    gap,
                    status,
                    app_now_text(),
                ),
            )

        conn.commit()


def get_questionnaire_by_id(questionnaire_id):
    if not questionnaire_id:
        return None, pd.DataFrame()

    with get_conn() as conn:
        questionnaire = conn.execute(
            """
            SELECT *
            FROM student_questionnaire
            WHERE questionnaire_id = ?
            LIMIT 1
            """,
            (questionnaire_id,),
        ).fetchone()
        if questionnaire is None:
            return None, pd.DataFrame()

        columns = [description[0] for description in conn.execute(
            "SELECT * FROM student_questionnaire LIMIT 0"
        ).description]
        questionnaire_dict = dict(zip(columns, questionnaire))
        courses_df = load_course_plan_for_questionnaire(conn, questionnaire_id)

    return questionnaire_dict, courses_df


def load_course_plan_for_questionnaire(conn, questionnaire_id):
    if not questionnaire_id:
        return pd.DataFrame()
    return pd.read_sql_query(
        """
        SELECT course_name, course_credits, weekly_study_hours,
               recommended_weekly_hours, study_gap, status
        FROM student_course_study_plan
        WHERE questionnaire_id = ?
        ORDER BY course_id
        """,
        conn,
        params=(questionnaire_id,),
    )


def find_questionnaire_id_for_prediction(conn, student_id, semester, prediction_time):
    normalized_student_id = normalize_student_id(student_id)
    normalized_semester = normalize_semester(semester)
    row = conn.execute(
        """
        SELECT questionnaire_id
        FROM student_questionnaire
        WHERE upper(student_id) = upper(?)
          AND semester = ?
          AND submitted_at <= ?
        ORDER BY submitted_at DESC, questionnaire_id DESC
        LIMIT 1
        """,
        (normalized_student_id, normalized_semester, prediction_time),
    ).fetchone()
    if row:
        return row[0]

    # Fallback for legacy questionnaire records saved before the semester field existed.
    row = conn.execute(
        """
        SELECT questionnaire_id
        FROM student_questionnaire
        WHERE upper(student_id) = upper(?)
          AND (semester IS NULL OR semester = '')
          AND submitted_at <= ?
        ORDER BY submitted_at DESC, questionnaire_id DESC
        LIMIT 1
        """,
        (normalized_student_id, prediction_time),
    ).fetchone()
    return row[0] if row else None


def get_questionnaire_for_prediction(record):
    if hasattr(record, "to_dict"):
        record = record.to_dict()

    questionnaire_id = record.get("questionnaire_id")
    if questionnaire_id and str(questionnaire_id).strip().lower() not in {"nan", "none", ""}:
        try:
            return get_questionnaire_by_id(int(float(questionnaire_id)))
        except Exception:
            pass

    student_id = record.get("student_id")
    semester = record.get("semester", "Semester 1")
    prediction_time = clean_display_value(record.get("prediction_time"))
    if not student_id or not prediction_time:
        return None, pd.DataFrame()

    with get_conn() as conn:
        matched_id = find_questionnaire_id_for_prediction(
            conn, student_id, semester, prediction_time
        )
    return get_questionnaire_by_id(matched_id)


def get_latest_questionnaire(student_id, semester=None):
    normalized_student_id = normalize_student_id(student_id)
    with get_conn() as conn:
        if semester:
            questionnaire = conn.execute(
                """
                SELECT *
                FROM student_questionnaire
                WHERE upper(student_id) = upper(?)
                  AND semester = ?
                ORDER BY submitted_at DESC, questionnaire_id DESC
                LIMIT 1
                """,
                (normalized_student_id, normalize_semester(semester)),
            ).fetchone()
        else:
            questionnaire = conn.execute(
                """
                SELECT *
                FROM student_questionnaire
                WHERE upper(student_id) = upper(?)
                ORDER BY submitted_at DESC, questionnaire_id DESC
                LIMIT 1
                """,
                (normalized_student_id,),
            ).fetchone()
        if questionnaire is None:
            return None, pd.DataFrame()

        columns = [description[0] for description in conn.execute(
            "SELECT * FROM student_questionnaire LIMIT 0"
        ).description]
        questionnaire_dict = dict(zip(columns, questionnaire))
        courses_df = load_course_plan_for_questionnaire(
            conn, questionnaire_dict["questionnaire_id"]
        )

    return questionnaire_dict, courses_df


def get_student_original_contact(student_id):
    normalized_student_id = normalize_student_id(student_id)
    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT student_id, student_name, student_email, department,
                   programme, phone_number, status, created_at
            FROM student_contacts
            WHERE upper(student_id) = upper(?)
            LIMIT 1
            """,
            (normalized_student_id,),
        ).fetchone()
    if row is None:
        return {}
    return {
        "student_id": row[0],
        "student_name": row[1],
        "student_email": row[2],
        "department": row[3],
        "programme": row[4],
        "phone_number": row[5],
        "status": row[6],
        "created_at": row[7],
    }


def questionnaire_completed(student_id):
    questionnaire, _ = get_latest_questionnaire(student_id)
    return questionnaire is not None


def questionnaire_completed_for_prediction(record):
    questionnaire, _ = get_questionnaire_for_prediction(record)
    return questionnaire is not None


def parse_json_list(value):
    if not value:
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [str(value)]


def normalize_yes_no(value):
    text = str(value).strip().lower()
    return "Yes" if text in {"yes", "y", "true", "1"} else "No"


SEMESTER_OPTIONS = [
    f"Semester {number}"
    for number in [
        "1",
        "1.5",
        "2",
        "2.5",
        "3",
        "3.5",
        "4",
        "4.5",
        "5",
        "5.5",
        "6",
        "6.5",
        "7",
        "7.5",
        "8",
    ]
]


def semester_numeric_value(value):
    text = str(value or "").strip()
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if not match:
        return 1.0
    number = float(match.group(1))
    return max(1.0, min(number, 8.0))


def normalize_semester(value):
    number = semester_numeric_value(value)
    if number.is_integer():
        display_number = str(int(number))
    else:
        display_number = f"{number:.1f}".rstrip("0").rstrip(".")
    return f"Semester {display_number}"


def semester_model_bucket(value):
    number = semester_numeric_value(value)
    return int(number)


def normalize_department(value):
    text = str(value).strip().lower()
    if "business" in text:
        return "Business"
    if "computer" in text or text == "cs" or "data" in text or "software" in text:
        return "CS"
    if "engineer" in text:
        return "Engineering"
    if "science" in text:
        return "Science"
    return "CS"


def normalize_stress(value):
    stress = float(value)
    if stress > 10:
        stress = stress / 10
    return max(0, min(stress, 10))


def make_input_dataframe(
    age,
    study_hours,
    attendance,
    assignment_delay,
    stress,
    internet,
    part_time_job,
    scholarship,
    semester,
    department,
):
    semester = normalize_semester(semester)
    semester_bucket = semester_model_bucket(semester)
    department = normalize_department(department)
    row = {
        "Age": float(age),
        "Study_Hours_per_Day": float(study_hours),
        "Attendance_Rate": float(attendance),
        "Assignment_Delay_Days": float(assignment_delay),
        "Stress_Index": normalize_stress(stress),
        "Internet_Access_Yes": 1 if normalize_yes_no(internet) == "Yes" else 0,
        "Part_Time_Job_Yes": 1 if normalize_yes_no(part_time_job) == "Yes" else 0,
        "Scholarship_Yes": 1 if normalize_yes_no(scholarship) == "Yes" else 0,
        "Semester_Year 2": 1 if semester_bucket == 2 else 0,
        "Semester_Year 3": 1 if semester_bucket == 3 else 0,
        "Semester_Year 4": 1 if semester_bucket == 4 else 0,
        "Department_Business": 1 if department == "Business" else 0,
        "Department_CS": 1 if department == "CS" else 0,
        "Department_Engineering": 1 if department == "Engineering" else 0,
        "Department_Science": 1 if department == "Science" else 0,
    }

    model_features = get_model_features()
    if model_features:
        return pd.DataFrame([row]).reindex(columns=model_features, fill_value=0)
    return pd.DataFrame([row])


def friendly_feature_name(feature):
    names = {
        "Age": "Age",
        "Study_Hours_per_Day": "Study Hours per Day",
        "Attendance_Rate": "Attendance Rate",
        "Assignment_Delay_Days": "Assignment Delay",
        "Stress_Index": "Stress Index",
        "Internet_Access_Yes": "Internet Access",
        "Part_Time_Job_Yes": "Part-Time Job",
        "Scholarship_Yes": "Scholarship",
        "Semester_Year 2": "Semester 2",
        "Semester_Year 3": "Semester 3",
        "Semester_Year 4": "Semester 4",
        "Department_Business": "Business Department",
        "Department_CS": "Computer Science Department",
        "Department_Engineering": "Engineering Department",
        "Department_Science": "Science Department",
    }
    return names.get(feature, feature)


def risk_label(probability):
    if probability >= 85:
        return "Critical Risk", "#8f1d1d", "Urgent intervention is recommended."
    if probability >= 70:
        return "High Risk", "#b4232f", "Immediate advisor monitoring is recommended."
    if probability >= 40:
        return "Medium Risk", "#8a5708", "Academic support and monitoring are recommended."
    return "Low Risk", "#17613a", "Student is currently stable."


def get_top_factors(input_data):
    try:
        model = get_prediction_model()
        explainer = get_prediction_explainer()
        if explainer is not None:
            shap_values = explainer.shap_values(input_data)
            if isinstance(shap_values, list):
                shap_row = shap_values[1][0]
            else:
                if len(shap_values.shape) == 3:
                    shap_row = shap_values[0, :, 1]
                else:
                    shap_row = shap_values[0]

            factors_df = pd.DataFrame(
                {"feature": input_data.columns, "impact": shap_row}
            )
            factors_df["abs_impact"] = factors_df["impact"].abs()
            factors_df = factors_df.sort_values("abs_impact", ascending=False).head(5)
            output = []
            for _, row in factors_df.iterrows():
                direction = "increased risk" if row["impact"] > 0 else "reduced risk"
                output.append(f"{friendly_feature_name(row['feature'])}: {direction}")
            return output

        if hasattr(model, "feature_importances_"):
            pairs = sorted(
                zip(input_data.columns, model.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:5]
            return [
                f"{friendly_feature_name(feature)}: important model factor"
                for feature, _ in pairs
            ]
    except Exception:
        pass

    return []


def predict_student(input_data):
    model = get_prediction_model()
    probability = float(model.predict_proba(input_data)[0][1]) * 100
    status, color, advice = risk_label(probability)
    factors = get_top_factors(input_data)
    return probability, status, color, advice, factors


def generate_interventions(
    risk_status,
    attendance,
    stress,
    assignment_delay,
    study_hours,
    gpa,
    internet,
    family_problems,
    part_time_job,
    factors,
):
    interventions = []

    if risk_status in {"Critical Risk", "High Risk"}:
        interventions.append("Schedule an advisor meeting within one week.")
        interventions.append("Create a weekly academic support plan.")
    elif risk_status == "Medium Risk":
        interventions.append("Monitor progress weekly for the next month.")
        interventions.append("Recommend tutoring or study-skills support.")
    else:
        interventions.append("Continue regular monitoring.")

    if attendance < 70:
        interventions.append("Improve attendance through check-ins and reminders.")
    if normalize_stress(stress) >= 7:
        interventions.append("Offer counselling or stress-management support.")
    if assignment_delay >= 5:
        interventions.append("Set smaller assignment milestones.")
    if study_hours < 3:
        interventions.append("Increase study hours gradually with a weekly target.")
    if gpa is not None and float(gpa) < 2.5:
        interventions.append("Recommend academic tutoring for low GPA.")
    if normalize_yes_no(internet) == "No":
        interventions.append("Provide internet or campus lab access support.")
    if normalize_yes_no(family_problems) == "Yes":
        interventions.append("Arrange a private advisor follow-up for family issues.")
    if normalize_yes_no(part_time_job) == "Yes":
        interventions.append("Review workload and study schedule balance.")
    if factors:
        interventions.append("Main model factors: " + "; ".join(factors[:3]))

    return interventions


def generate_ai_suggestions(risk_status, probability, factors, interventions):
    prompt = f"""
You are a university academic advisor.

Create a professional student intervention report.

Student Risk Level:
{risk_status}

Dropout Probability:
{probability:.2f}%

Main SHAP Risk Factors:
{factors}

Current Interventions:
{interventions}

Generate the response using EXACTLY these sections:

### Key Risk Indicators
### Suggested Interventions
### Advisor Follow-Up Actions

Rules:
- Use bullet points
- Keep it professional
- Keep it short and practical
- Treat the prediction as an early-warning signal, not a certainty
"""
    success, result = call_gemini_api(prompt)
    if success:
        return result
    return (
        "### Gemini AI Advisor unavailable\n\n"
        f"{result}\n\n"
        "Add or correct the Gemini key, restart AURA, and run the prediction again."
    )


def save_prediction(
    input_method,
    student_name,
    student_id,
    age,
    study_hours,
    attendance,
    assignment_delay,
    stress,
    gpa,
    internet,
    part_time_job,
    family_problems,
    family_reason,
    scholarship,
    department,
    semester,
    probability,
    risk_status,
    factors,
    interventions,
    ai_suggestions,
    upload_batch_id="",
):
    with get_conn() as conn:
        cursor = conn.cursor()
        prediction_time = app_now_text()
        normalized_semester = normalize_semester(semester)
        matched_questionnaire_id = find_questionnaire_id_for_prediction(
            conn, student_id, normalized_semester, prediction_time
        )
        cursor.execute(
            """
            INSERT INTO student_records (
                uploaded_by, student_name, student_id, age, study_hours,
                attendance_rate, assignment_delay, semester, gpa, internet_access,
                part_time_job, family_problems, family_reason, scholarship,
                department, stress_index, uploaded_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                st.session_state.user_email,
                student_name,
                student_id,
                age,
                study_hours,
                attendance,
                assignment_delay,
                normalized_semester,
                gpa,
                internet,
                part_time_job,
                family_problems,
                family_reason,
                scholarship,
                department,
                normalize_stress(stress),
                prediction_time,
            ),
        )
        cursor.execute(
            """
            INSERT INTO prediction_history (
                student_id, student_name, predicted_risk, probability_score,
                semester, questionnaire_id, gpa, stress_index, predicted_by, input_method,
                upload_batch_id, top_factors, interventions, ai_suggestions,
                prediction_time
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                student_id,
                student_name,
                risk_status,
                probability,
                normalized_semester,
                matched_questionnaire_id,
                gpa,
                normalize_stress(stress),
                st.session_state.user_email,
                input_method,
                upload_batch_id,
                "; ".join(factors),
                json.dumps(interventions),
                ai_suggestions,
                prediction_time,
            ),
        )
        prediction_id = cursor.lastrowid
        notification_record = {
            "prediction_id": prediction_id,
            "upload_batch_id": upload_batch_id,
            "student_id": student_id,
            "student_name": student_name,
            "predicted_risk": risk_status,
            "probability_score": round(probability, 2),
            "top_factors": "; ".join(factors),
        }
        if matched_questionnaire_id:
            questionnaire, course_plan_df = get_questionnaire_by_id(matched_questionnaire_id)
            subject, body = suggested_notification_message(
                notification_record, questionnaire, course_plan_df, "Initial Alert"
            )
            notification_record["message_subject"] = subject
            notification_record["message_body"] = body
        notifications.ensure_pending_initial_notification(
            conn,
            notification_record,
        )
        cursor.execute(
            """
            INSERT INTO audit_logs (
                user_email, user_role, action_type, action_status, action_details,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                st.session_state.user_email,
                st.session_state.user_role,
                input_method.upper(),
                "SUCCESS",
                (
                    f"Predicted risk for {student_name} ({student_id}); "
                    f"questionnaire snapshot: {matched_questionnaire_id or 'not available'}"
                ),
                prediction_time,
            ),
        )
        conn.commit()
        return prediction_id


def load_history():
    with get_conn() as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM prediction_history
            ORDER BY prediction_time DESC
            """,
            conn,
        )


def is_administrator_user():
    return st.session_state.user_role in {"admin", "administrator"}


def is_advisor_user():
    return st.session_state.user_role == "advisor"


def can_send_notifications():
    return is_advisor_user()


def get_email_settings():
    return notifications.email_settings_from_secrets(st.secrets)


def fetch_prediction_record(prediction_id):
    if not prediction_id:
        return None
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM prediction_history WHERE prediction_id = ?",
            (prediction_id,),
        ).fetchone()
        if row is None:
            return None
        columns = [description[0] for description in conn.execute(
            "SELECT * FROM prediction_history LIMIT 0"
        ).description]
    return dict(zip(columns, row))


def notification_status_for_record(row):
    prediction_id = row.get("prediction_id") if hasattr(row, "get") else None
    risk_level = row.get("predicted_risk") if hasattr(row, "get") else ""
    if not prediction_id:
        return "Not Required" if not notifications.is_notifiable_risk(risk_level) else "Pending"
    with get_conn() as conn:
        return notifications.latest_notification_status(conn, prediction_id, risk_level)


def load_upload_batches():
    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT upload_batch_id
            FROM prediction_history
            WHERE upload_batch_id IS NOT NULL AND upload_batch_id != ''
            ORDER BY upload_batch_id DESC
            """,
            conn,
        )
    return df["upload_batch_id"].tolist() if not df.empty else []


def update_notification_log(notification_id, status, status_message="", sent_by=None):
    with get_conn() as conn:
        conn.execute(
            """
            UPDATE notification_logs
            SET send_status = ?, status_message = ?, sent_by = ?,
                sent_at = CASE WHEN ? IN ('Sent', 'Resent', 'Failed') THEN ? ELSE sent_at END
            WHERE notification_id = ?
            """,
            (
                status,
                status_message,
                sent_by or st.session_state.user_email,
                status,
                app_now_text(),
                notification_id,
            ),
        )
        conn.commit()


def send_notification_by_id(notification_id, resend=False):
    if not can_send_notifications():
        return False, "Only Academic Advisors can send notifications."

    with get_conn() as conn:
        row = conn.execute(
            """
            SELECT notification_id, student_email, message_subject, message_body,
                   send_status
            FROM notification_logs
            WHERE notification_id = ?
            """,
            (notification_id,),
        ).fetchone()
    if row is None:
        return False, "Notification record not found."

    _, student_email, subject, body, current_status = row
    if current_status == "Sent" and not resend:
        return False, "This notification has already been sent."

    sent, message = notifications.send_email(
        get_email_settings(),
        student_email,
        subject,
        body,
    )
    if sent:
        update_notification_log(
            notification_id,
            "Resent" if resend else "Sent",
            message,
            st.session_state.user_email,
        )
    else:
        update_notification_log(
            notification_id,
            "Failed",
            message,
            st.session_state.user_email,
        )
    return sent, message


BULK_NOTIFICATION_SEND_LIMIT = 5


def is_smtp_rate_limit_message(message):
    text = str(message or "").lower()
    return (
        "too many messages" in text
        or "rate limit" in text
        or "quota" in text
        or "554" in text
    )


def send_pending_notifications_for_batch(upload_batch_id):
    if not can_send_notifications():
        return 0, 0, 0, ["Only Academic Advisors can send notifications."]
    if not upload_batch_id:
        return 0, 0, 0, ["Please select one upload batch first."]

    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT notification_id
            FROM notification_logs
            WHERE upload_batch_id = ?
              AND notification_type = 'Initial Alert'
              AND send_status IN ('Pending', 'Failed')
            ORDER BY notification_id
            """,
            (upload_batch_id,),
        ).fetchall()

    sent_count = 0
    failed_count = 0
    messages = []
    selected_rows = rows[:BULK_NOTIFICATION_SEND_LIMIT]
    remaining_count = max(0, len(rows) - len(selected_rows))
    for row in selected_rows:
        sent, message = send_notification_by_id(row[0], resend=False)
        if sent:
            sent_count += 1
        else:
            failed_count += 1
            messages.append(message)
            if is_smtp_rate_limit_message(message):
                remaining_count += len(selected_rows) - sent_count - failed_count
                messages.append(
                    "Bulk sending stopped because the SMTP server rate limit was reached. "
                    "Please wait before sending more alerts."
                )
                break
    return sent_count, failed_count, remaining_count, messages


def ensure_pending_notifications_for_df(df):
    if df.empty or "prediction_id" not in df.columns:
        return
    with get_conn() as conn:
        for _, row in df.iterrows():
            risk_level = normalize_risk_filter_value(row.get("predicted_risk"))
            if not notifications.is_notifiable_risk(risk_level):
                continue
            notifications.ensure_pending_initial_notification(
                conn,
                {
                    "prediction_id": row.get("prediction_id"),
                    "upload_batch_id": row.get("upload_batch_id", ""),
                    "student_id": row.get("student_id", ""),
                    "student_name": row.get("student_name", ""),
                    "predicted_risk": risk_level,
                    "probability_score": row.get("probability_score", ""),
                    "top_factors": row.get("top_factors", ""),
                },
            )
        conn.commit()


def load_notification_history(prediction_id=None, student_id=None):
    with get_conn() as conn:
        if prediction_id:
            return pd.read_sql_query(
                """
                SELECT notification_type AS Type, risk_level AS "Risk Level",
                       send_status AS Status, sent_at AS "Sent Time",
                       sent_by AS "Sent By", status_message AS "Message"
                FROM notification_logs
                WHERE prediction_id = ?
                ORDER BY COALESCE(sent_at, created_at) DESC, notification_id DESC
                """,
                conn,
                params=(prediction_id,),
            )
        return pd.read_sql_query(
            """
            SELECT notification_type AS Type, risk_level AS "Risk Level",
                   send_status AS Status, sent_at AS "Sent Time",
                   sent_by AS "Sent By", status_message AS "Message"
            FROM notification_logs
            WHERE upper(student_id) = upper(?)
            ORDER BY COALESCE(sent_at, created_at) DESC, notification_id DESC
            """,
            conn,
            params=(student_id,),
        )


def get_latest_notification_detail(prediction_id):
    with get_conn() as conn:
        row = notifications.latest_notification_for_prediction(conn, prediction_id)
    if row is None:
        return None
    return {
        "notification_id": row[0],
        "notification_type": row[1],
        "send_status": row[2],
        "sent_at": row[3],
        "sent_by": row[4],
        "message_subject": row[5],
        "message_body": row[6],
        "student_email": row[7],
    }


def create_notification_log(record, notification_type, subject, body):
    with get_conn() as conn:
        student_email = notifications.find_student_email(conn, record.get("student_id"))
        cursor = conn.execute(
            """
            INSERT INTO notification_logs (
                upload_batch_id, prediction_id, student_id, student_email, risk_level,
                notification_type, message_subject, message_body, send_status,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending', ?)
            """,
            (
                record.get("upload_batch_id", ""),
                record.get("prediction_id"),
                record.get("student_id", ""),
                student_email,
                normalize_risk_filter_value(record.get("predicted_risk")),
                notification_type,
                subject,
                body,
                app_now_text(),
            ),
        )
        conn.commit()
        return cursor.lastrowid


def suggested_notification_message(record, questionnaire, courses_df, notification_type="Initial Alert"):
    course_recommendations = course_study_recommendations(courses_df)
    return notifications.build_system_message(
        record,
        questionnaire=questionnaire,
        course_recommendations=course_recommendations,
        notification_type=notification_type,
    )


def send_new_notification(record, notification_type, subject, body):
    if not can_send_notifications():
        return False, "Only Academic Advisors can send notifications."
    notification_id = create_notification_log(record, notification_type, subject, body)
    return send_notification_by_id(notification_id, resend=notification_type != "Initial Alert")


def load_audit_logs():
    with get_conn() as conn:
        return pd.read_sql_query(
            """
            SELECT log_id, created_at, user_email, user_role,
                   action_type, action_status, action_details
            FROM audit_logs
            ORDER BY log_id DESC
            """,
            conn,
        )


def normalize_audit_action(value):
    text = " ".join(clean_display_value(value, "Unknown").replace("_", " ").split())
    normalized = text.lower()
    action_labels = {
        "manual prediction": "Manual Prediction",
        "upload csv": "Upload CSV",
        "login": "Login",
        "logout": "Logout",
        "register": "Register",
        "create user": "Create User",
        "update role": "Update Role",
    }
    return action_labels.get(normalized, text.title())


def normalize_audit_status(value):
    text = " ".join(clean_display_value(value, "Fail").split()).lower()
    if text in {"success", "successful", "yes", "true", "1", "ok"}:
        return "Success"
    if text in {"failed", "failure", "error", "no", "false", "0"}:
        return "Fail"
    return "Success" if text else "Fail"


def show_result(probability, risk_status, color, advice, factors, interventions):
    st.markdown(
        f"""
        <div class="risk-card" style="background-color:{color};">
            <h3>Dropout Risk Probability: {probability:.2f}%</h3>
            <h4>Status: {risk_status}</h4>
            <p>{advice}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.subheader("Top Model Factors")
    if factors:
        for factor in factors:
            st.write(f"- {factor}")
    else:
        st.info("Model factor explanation is not available.")

    st.subheader("Recommended Interventions")
    for item in interventions:
        st.write(f"- {item}")


def clean_display_value(value, fallback=""):
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def safe_html(value, fallback=""):
    return html.escape(clean_display_value(value, fallback))


def clean_display_number(value, fallback=0.0):
    try:
        if value is None or pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def normalize_risk_filter_value(value):
    text = clean_display_value(value).lower()
    text = " ".join(text.split())
    risk_labels = {
        "critical risk": "Critical Risk",
        "high risk": "High Risk",
        "medium risk": "Medium Risk",
        "low risk": "Low Risk",
        "safe": "Low Risk",
    }
    return risk_labels.get(text, clean_display_value(value))


def split_detail_items(value):
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    text = clean_display_value(value)
    if not text:
        return []

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except (json.JSONDecodeError, TypeError):
        pass

    for separator in [" | ", "; "]:
        if separator in text:
            return [item.strip() for item in text.split(separator) if item.strip()]

    return [text]


def risk_display_style(risk_status, probability):
    colors = {
        "Critical Risk": "#8f1d1d",
        "High Risk": "#b4232f",
        "Medium Risk": "#8a5708",
        "Low Risk": "#17613a",
        "Failed": "#4b5563",
    }
    advice = {
        "Critical Risk": "Urgent intervention is recommended.",
        "High Risk": "Immediate advisor monitoring is recommended.",
        "Medium Risk": "Academic support and monitoring are recommended.",
        "Low Risk": "Student is currently stable.",
        "Failed": "Prediction failed for this student.",
    }
    if risk_status in colors:
        return colors[risk_status], advice[risk_status]
    _, color, message = risk_label(probability)
    return color, message


def dashboard_risk_label(value):
    risk_status = normalize_risk_filter_value(value)
    if risk_status == "Critical Risk":
        return "High Risk"
    return risk_status


def dashboard_risk_cell_style(value):
    colors = {
        "High Risk": "#b4232f",
        "Medium Risk": "#8a5708",
        "Low Risk": "#17613a",
    }
    color = colors.get(str(value), "#4b5563")
    return (
        f"background-color: {color}; color: white; font-weight: 700; "
        "text-align: center;"
    )


def dashboard_recent_row_style(row):
    return [
        dashboard_risk_cell_style(row["predicted_risk"])
        if column == "predicted_risk"
        else ""
        for column in row.index
    ]


def first_display_value(record, keys, fallback=""):
    for key in keys:
        value = clean_display_value(record.get(key))
        if value:
            return value
    return fallback


def load_student_record_detail(student_id):
    if not student_id or student_id == "Unknown ID":
        return {}

    try:
        with get_conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT *
                FROM student_records
                WHERE student_id = ?
                ORDER BY uploaded_at DESC
                LIMIT 1
                """,
                conn,
                params=(student_id,),
            )
    except Exception:
        return {}

    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def render_risk_badge(risk_status, probability, show_probability=False):
    color, _ = risk_display_style(risk_status, probability)
    probability_line = ""
    if show_probability:
        probability_line = (
            f'<br><span style="font-size:13px; font-weight:500;">'
            f"{probability:.2f}%</span>"
        )
    st.markdown(
        f"""
        <div class="student-risk-badge" style="background:{color};">
            {safe_html(risk_status)}{probability_line}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_questionnaire_badge(is_completed):
    label = "Completed" if is_completed else "Not Completed"
    css_class = "questionnaire-completed" if is_completed else "questionnaire-missing"
    st.markdown(
        f"""
        <div class="questionnaire-badge {css_class}">
            {safe_html(label)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_notification_badge(status):
    status = clean_display_value(status, "Not Required")
    css_key = status.lower().replace(" ", "-")
    st.markdown(
        f"""
        <div class="notification-badge notification-{safe_html(css_key)}">
            {safe_html(status)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_manual_prediction_result(
    student_name,
    student_id,
    semester,
    gpa,
    stress,
    probability,
    risk_status,
    advice,
):
    status_key = str(risk_status or "").strip().lower().replace(" risk", "")
    if status_key not in {"critical", "high", "medium", "low"}:
        status_key = "high"

    facts = [
        ("Student", student_name),
        ("Student ID", student_id),
        ("Semester", semester),
        ("GPA", f"{float(gpa):.2f}"),
        ("Stress Index", f"{normalize_stress(stress):.1f} / 10"),
    ]
    fact_html = "".join(
        '<div class="prediction-fact">'
        f'<div class="prediction-fact-label">{safe_html(label)}</div>'
        f'<div class="prediction-fact-value">{safe_html(value, "N/A")}</div>'
        "</div>"
        for label, value in facts
    )

    st.markdown(
        f"""
        <div class="prediction-overview prediction-overview-{status_key}">
            <div>
                <div class="prediction-overview-eyebrow">Student risk assessment</div>
                <div class="prediction-overview-title">{safe_html(risk_status)}</div>
                <div class="prediction-overview-copy">{safe_html(advice)}</div>
            </div>
            <div class="prediction-overview-score">
                <strong>{float(probability):.1f}%</strong>
                <span>Dropout probability</span>
            </div>
        </div>
        <div class="prediction-facts">{fact_html}</div>
        """,
        unsafe_allow_html=True,
    )


def render_detail_grid(items):
    cards = []
    for label, value in items:
        cards.append(
            '<div class="detail-item">'
            f'<div class="detail-label">{safe_html(label)}</div>'
            f'<div class="detail-value">{safe_html(value, "N/A")}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="detail-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def render_detail_list_card(items):
    bullet_items = "".join(f"<li>{safe_html(item)}</li>" for item in items)
    st.markdown(
        f'<div class="detail-list-card"><ul>{bullet_items}</ul></div>',
        unsafe_allow_html=True,
    )


def questionnaire_summary_items(questionnaire):
    if not questionnaire:
        return []
    return [
        ("Questionnaire Status", "Completed"),
        ("Questionnaire Semester", clean_display_value(questionnaire.get("semester"), "N/A")),
        ("Last Updated", clean_display_value(questionnaire.get("submitted_at"), "N/A")),
        ("Preferred Alert Method", ", ".join(parse_json_list(questionnaire.get("alert_method"))) or "N/A"),
        ("Preferred Update Frequency", clean_display_value(questionnaire.get("update_frequency"), "N/A")),
        ("Preferred Support", ", ".join(parse_json_list(questionnaire.get("preferred_support"))) or "N/A"),
        ("Main Communication Barrier", ", ".join(parse_json_list(questionnaire.get("main_barrier"))) or "N/A"),
        ("Communication Ease", f"{clean_display_value(questionnaire.get('communication_ease'), 'N/A')} / 5"),
        ("Allow Risk Flagging", clean_display_value(questionnaire.get("allow_risk_flagging"), "N/A")),
        ("High-Risk Reminder Consent", clean_display_value(questionnaire.get("high_risk_reminder_consent"), "N/A")),
        ("Preferred Contact Method", ", ".join(parse_json_list(questionnaire.get("preferred_contact_method"))) or "N/A"),
        ("Preferred Contact Details", clean_display_value(questionnaire.get("preferred_contact_details"), "N/A")),
    ]


def course_study_recommendations(courses_df):
    if courses_df.empty:
        return []

    recommendations = []
    for _, course in courses_df.iterrows():
        name = clean_display_value(course.get("course_name"), "Course")
        gap = clean_display_number(course.get("study_gap"))
        required = clean_display_number(course.get("recommended_weekly_hours"))
        if gap > 0:
            recommendations.append(
                f"Increase {name} study time by {gap:.1f} hours per week "
                f"(recommended: {required:.1f} hours/week)."
            )
        else:
            recommendations.append(f"{name} is on track for weekly study time.")
    return recommendations


def build_personalized_advisor_note(record, questionnaire, courses_df):
    if not questionnaire:
        return ""

    risk_status = normalize_risk_filter_value(record.get("predicted_risk"))
    preferred_support = parse_json_list(questionnaire.get("preferred_support"))
    contact_methods = parse_json_list(questionnaire.get("preferred_contact_method"))
    alert_methods = parse_json_list(questionnaire.get("alert_method"))
    barriers = parse_json_list(questionnaire.get("main_barrier"))
    frequency = clean_display_value(questionnaire.get("update_frequency"), "regular")
    high_risk_reminder = clean_display_value(
        questionnaire.get("high_risk_reminder_consent"), "N/A"
    )

    contact = ", ".join(contact_methods or alert_methods) or "the student's preferred contact channel"
    support = ", ".join(preferred_support) or "academic support"
    barrier = ", ".join(barriers) or "no major communication barrier reported"
    course_gaps = [
        clean_display_value(row.get("course_name"), "Course")
        for _, row in courses_df.iterrows()
        if clean_display_number(row.get("study_gap")) > 0
    ]

    note = (
        f"Based on the latest student questionnaire, this {risk_status.lower()} case "
        f"should be handled with {support}. The student prefers contact through "
        f"{contact} and would like {frequency.lower()} updates. Main reported barrier: "
        f"{barrier}."
    )
    if course_gaps:
        note += " Course-level study time is below the recommended level for: " + ", ".join(course_gaps) + "."
    if high_risk_reminder != "N/A":
        note += f" High-risk reminder consent: {high_risk_reminder}."
    return note


def build_personalized_interventions(questionnaire, courses_df):
    if not questionnaire:
        return []

    interventions = []
    contact_methods = parse_json_list(questionnaire.get("preferred_contact_method"))
    alert_methods = parse_json_list(questionnaire.get("alert_method"))
    preferred_support = parse_json_list(questionnaire.get("preferred_support"))
    barriers = parse_json_list(questionnaire.get("main_barrier"))
    frequency = clean_display_value(questionnaire.get("update_frequency"))
    high_risk_reminder = clean_display_value(
        questionnaire.get("high_risk_reminder_consent")
    )

    contact = ", ".join(contact_methods or alert_methods)
    if contact:
        interventions.append(f"Contact the student through {contact}.")
    if preferred_support:
        interventions.append("Start with preferred support: " + ", ".join(preferred_support) + ".")
    if frequency:
        interventions.append(f"Send progress updates {frequency.lower()}.")
    if high_risk_reminder == "Yes":
        interventions.append("Send a reminder if AURA detects the student as high risk.")
    if barriers:
        interventions.append("Address communication barrier: " + ", ".join(barriers) + ".")
    interventions.extend(course_study_recommendations(courses_df))
    return interventions


def original_contact_items(original_contact):
    if not original_contact:
        return [("Original Contact", "No school contact record found")]
    return [
        ("Original Student Email", clean_display_value(original_contact.get("student_email"), "N/A")),
        ("Original Phone Number", clean_display_value(original_contact.get("phone_number"), "N/A")),
        ("Programme", clean_display_value(original_contact.get("programme"), "N/A")),
        ("Contact Status", clean_display_value(original_contact.get("status"), "N/A")),
    ]


def render_support_profile(questionnaire, courses_df, original_contact=None):
    original_contact = original_contact or {}
    if not questionnaire:
        st.info(
            "Questionnaire Status: Not Completed. This student has not completed "
            "the student support questionnaire yet. Advisor recommendations are "
            "currently based only on academic prediction data."
        )
        st.subheader("School Original Contact Information")
        render_detail_grid(original_contact_items(original_contact))
        return

    render_detail_grid(questionnaire_summary_items(questionnaire))
    st.subheader("School Original Contact Information")
    render_detail_grid(original_contact_items(original_contact))
    st.subheader("Course Study Load")
    if courses_df.empty:
        st.info("No course study load was submitted.")
    else:
        st.dataframe(courses_df, use_container_width=True)
        recommendations = course_study_recommendations(courses_df)
        if recommendations:
            st.subheader("Course Study Recommendations")
            render_detail_list_card(recommendations)


def render_prediction_detail(record):
    if hasattr(record, "to_dict"):
        record = record.to_dict()

    student_name = clean_display_value(record.get("student_name"), "Unknown Student")
    student_id = clean_display_value(record.get("student_id"), "Unknown ID")
    stored_details = load_student_record_detail(student_id)
    for key, value in stored_details.items():
        record.setdefault(key, value)
    questionnaire, course_plan_df = get_questionnaire_for_prediction(record)
    original_contact = get_student_original_contact(student_id)
    has_questionnaire = questionnaire is not None

    risk_status = normalize_risk_filter_value(record.get("predicted_risk"))
    probability = clean_display_number(record.get("probability_score"))
    color, advice = risk_display_style(risk_status, probability)

    if risk_status == "Failed":
        st.error(clean_display_value(record.get("error"), "Prediction failed for this student."))
    else:
        st.markdown(
            f"""
            <div class="detail-hero">
                <div>
                    <div class="detail-eyebrow">Student Prediction Result</div>
                    <h2>{safe_html(student_name)}</h2>
                    <div class="detail-subtitle">
                        Student ID: {safe_html(student_id)} | {safe_html(advice)}
                    </div>
                </div>
                <div class="detail-status" style="background:{color};">
                    {safe_html(risk_status)}
                    <span>{probability:.2f}% dropout probability</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    details_tab, support_tab, advisor_tab, interventions_tab, factors_tab, notification_tab = st.tabs(
        [
            "Student Details",
            "Support Profile",
            "AI Advisor Note",
            "Interventions",
            "SHAP Factors",
            "Notification",
        ]
    )

    with details_tab:
        detail_items = [
            ("Student Name", student_name),
            ("Student ID", student_id),
            ("Questionnaire", "Completed" if has_questionnaire else "Not Completed"),
            ("Risk Level", risk_status),
            ("Dropout Probability", f"{probability:.2f}%"),
            ("Age", first_display_value(record, ["age"], "N/A")),
            ("Department", first_display_value(record, ["department"], "N/A")),
            ("Semester", first_display_value(record, ["semester"], "N/A")),
            ("GPA", first_display_value(record, ["gpa"], "N/A")),
            ("Study Hours per Day", first_display_value(record, ["study_hours"], "N/A")),
            (
                "Attendance Rate",
                first_display_value(record, ["attendance", "attendance_rate"], "N/A"),
            ),
            (
                "Assignment Delay Days",
                first_display_value(record, ["assignment_delay"], "N/A"),
            ),
            (
                "Stress Index",
                first_display_value(record, ["stress_index", "stress"], "N/A"),
            ),
            (
                "Internet Access",
                first_display_value(record, ["internet", "internet_access"], "N/A"),
            ),
            ("Part-Time Job", first_display_value(record, ["part_time_job"], "N/A")),
            ("Family Problems", first_display_value(record, ["family_problems"], "N/A")),
            ("Scholarship", first_display_value(record, ["scholarship"], "N/A")),
        ]

        input_method = clean_display_value(record.get("input_method"))
        if input_method:
            detail_items.append(("Input Method", input_method))

        predicted_by = clean_display_value(record.get("predicted_by"))
        if predicted_by:
            detail_items.append(("Predicted By", predicted_by))

        prediction_time = clean_display_value(record.get("prediction_time"))
        if prediction_time:
            detail_items.append(("Prediction Time", prediction_time))

        family_reason = first_display_value(record, ["family_reason"])
        if family_reason:
            detail_items.append(("Family Problem Note", family_reason))

        render_detail_grid(detail_items)
        show_chat_fab("details")

    with support_tab:
        render_support_profile(questionnaire, course_plan_df, original_contact)
        show_chat_fab("support")

    with advisor_tab:
        personalized_note = build_personalized_advisor_note(
            record, questionnaire, course_plan_df
        )
        if personalized_note:
            st.markdown("### Personalized Support Note")
            st.write(personalized_note)
        ai_suggestions = clean_display_value(record.get("ai_suggestions"))
        if ai_suggestions:
            if personalized_note:
                st.markdown("### Academic Risk Note")
            st.markdown(ai_suggestions)
        elif not personalized_note:
            st.info("No AI advisor note available.")
        show_chat_fab("advisor")

    with interventions_tab:
        personalized_interventions = build_personalized_interventions(
            questionnaire, course_plan_df
        )
        if personalized_interventions:
            st.markdown("### Personalized Actions")
            render_detail_list_card(personalized_interventions)
        interventions = split_detail_items(record.get("interventions"))
        if interventions:
            if personalized_interventions:
                st.markdown("### Academic Risk Actions")
            render_detail_list_card(interventions)
        elif not personalized_interventions:
            st.info("No intervention suggestions available.")
        show_chat_fab("interventions")

    with factors_tab:
        factors = split_detail_items(record.get("top_factors"))
        if factors:
            render_detail_list_card(factors)
        else:
            st.info("No SHAP factors available.")
        show_chat_fab("factors")

    with notification_tab:
        prediction_id = clean_display_value(record.get("prediction_id"))
        latest_notification = get_latest_notification_detail(prediction_id)
        status = notification_status_for_record(record)
        status_items = [
            ("Latest Notification", status),
            (
                "Last Sent Time",
                clean_display_value(
                    latest_notification.get("sent_at") if latest_notification else "",
                    "N/A",
                ),
            ),
            (
                "Notification Type",
                clean_display_value(
                    latest_notification.get("notification_type") if latest_notification else "",
                    "Initial Alert" if notifications.is_notifiable_risk(risk_status) else "Not Required",
                ),
            ),
            (
                "Sent By",
                clean_display_value(
                    latest_notification.get("sent_by") if latest_notification else "",
                    "N/A",
                ),
            ),
        ]
        render_detail_grid(status_items)
        

        if not notifications.is_notifiable_risk(risk_status):
            st.info("Notifications are not required for Low Risk or Medium Risk students.")
        elif not can_send_notifications():
            st.warning("Only Academic Advisors can send student notifications.")
        else:
            st.subheader("Semi-Automatic Message")
            suggested_subject, suggested_body = suggested_notification_message(
                record, questionnaire, course_plan_df, "Initial Alert"
            )
            st.text_input(
                "Suggested Subject",
                value=suggested_subject,
                key=f"notification_suggested_subject_{prediction_id}",
                disabled=True,
            )
            st.text_area(
                "Suggested Message",
                value=suggested_body,
                height=260,
                key=f"notification_suggested_body_{prediction_id}",
                disabled=True,
            )
            if st.button(
                "Send System Suggested Alert",
                use_container_width=True,
                type="primary",
            ):
                notification_id = None
                resend_existing = False
                with get_conn() as conn:
                    row = conn.execute(
                        """
                        SELECT notification_id, send_status
                        FROM notification_logs
                        WHERE prediction_id = ?
                          AND notification_type = 'Initial Alert'
                        ORDER BY
                            CASE send_status
                                WHEN 'Pending' THEN 1
                                WHEN 'Failed' THEN 2
                                WHEN 'Sent' THEN 3
                                WHEN 'Resent' THEN 4
                                ELSE 5
                            END,
                            notification_id DESC
                        LIMIT 1
                        """,
                        (prediction_id,),
                    ).fetchone()
                    if row:
                        notification_id = row[0]
                        resend_existing = row[1] in {"Sent", "Resent"}
                    else:
                        notification_id = notifications.ensure_pending_initial_notification(
                            conn, record
                        )
                        conn.commit()
                if notification_id:
                    sent, message = send_notification_by_id(
                        notification_id,
                        resend=resend_existing,
                    )
                    st.success(message) if sent else st.error(message)
                    st.rerun()
                else:
                    st.error("No notification record is available for this student.")

            preferred_methods = parse_json_list(
                questionnaire.get("preferred_contact_method") if questionnaire else ""
            )
            contact_details = clean_display_value(
                questionnaire.get("preferred_contact_details") if questionnaire else ""
            )
            if any("WhatsApp" in method for method in preferred_methods):
                st.link_button(
                    "Open WhatsApp Message",
                    notifications.whatsapp_link(contact_details, suggested_body),
                    use_container_width=True,
                )

            st.subheader("Manual Custom Message")
            custom_subject = st.text_input(
                "Subject",
                value="AURA Academic Support Message",
                key=f"custom_notification_subject_{prediction_id}",
            )
            custom_body = st.text_area(
                "Message Body",
                value=suggested_body,
                height=260,
                key=f"custom_notification_body_{prediction_id}",
            )
            if st.button("Send Custom Message", use_container_width=True, type="primary"):
                sent, message = send_new_notification(
                    record,
                    "Custom Advisor Message",
                    custom_subject,
                    custom_body,
                )
                st.success(message) if sent else st.error(message)
                st.rerun()

        st.subheader("Notification History")
        history_df = load_notification_history(
            prediction_id=prediction_id,
            student_id=student_id,
        )
        if history_df.empty:
            st.info("No notification history yet.")
        else:
            st.dataframe(history_df, use_container_width=True)
        show_chat_fab("notification")


def prediction_record_id(row, index):
    if "prediction_id" in row:
        record_id = clean_display_value(row.get("prediction_id"))
        if record_id:
            return record_id
    return str(index)


def render_student_prediction_list(df, key_prefix, page_size=20):
    if df.empty:
        st.info("No students found.")
        return

    list_df = df.reset_index(drop=True)
    total_students = len(list_df)
    page_size = max(1, int(page_size))
    total_pages = max(1, (total_students + page_size - 1) // page_size)
    page_key = f"{key_prefix}_page"
    current_page = int(st.session_state.get(page_key, 1))
    current_page = min(max(current_page, 1), total_pages)
    st.session_state[page_key] = current_page
    start_index = (current_page - 1) * page_size
    end_index = min(start_index + page_size, total_students)
    page_df = list_df.iloc[start_index:end_index]

    st.subheader("Students")
    st.markdown(
        f"""
        <div class="student-list-meta">
            Showing students {start_index + 1}-{end_index} of {total_students}.
        </div>
        """,
        unsafe_allow_html=True,
    )

    header_cols = st.columns([3, 2, 2, 1.6, 1.6, 1.7, 1])
    header_cols[0].markdown('<div class="student-header-cell">Student Name</div>', unsafe_allow_html=True)
    header_cols[1].markdown('<div class="student-header-cell">Student ID</div>', unsafe_allow_html=True)
    header_cols[2].markdown('<div class="student-header-cell">Prediction Result</div>', unsafe_allow_html=True)
    header_cols[3].markdown('<div class="student-header-cell">Questionnaire</div>', unsafe_allow_html=True)
    header_cols[4].markdown('<div class="student-header-cell">Risk Probability</div>', unsafe_allow_html=True)
    header_cols[5].markdown('<div class="student-header-cell">Notification Status</div>', unsafe_allow_html=True)
    header_cols[6].markdown('<div class="student-header-cell">View</div>', unsafe_allow_html=True)

    for index, row in page_df.iterrows():
        record_id = prediction_record_id(row, index)
        student_name = clean_display_value(row.get("student_name"), "Unknown Student")
        student_id = clean_display_value(row.get("student_id"), "Unknown ID")
        prediction_time = clean_display_value(row.get("prediction_time"))
        risk_status = normalize_risk_filter_value(row.get("predicted_risk"))
        probability = clean_display_number(row.get("probability_score"))
        has_questionnaire = questionnaire_completed_for_prediction(row)
        notification_status = notification_status_for_record(row)

        name_col, id_col, result_col, questionnaire_col, probability_col, notification_col, action_col = st.columns(
            [3, 2, 2, 1.6, 1.6, 1.7, 1]
        )
        with name_col:
            prediction_line = ""
            if prediction_time:
                prediction_line = (
                    f'<div class="student-row-subtle">Prediction: '
                    f"{safe_html(prediction_time)}</div>"
                )
            st.markdown(
                f"""
                <div class="student-row-cell">
                    <div class="student-row-primary">{safe_html(student_name)}</div>
                    {prediction_line}
                </div>
                """,
                unsafe_allow_html=True,
            )
        with id_col:
            st.markdown(
                f"""
                <div class="student-row-cell">
                    <div class="student-row-primary">{safe_html(student_id)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with result_col:
            st.markdown('<div style="height:0.55rem;"></div>', unsafe_allow_html=True)
            render_risk_badge(risk_status, probability)
        with questionnaire_col:
            st.markdown('<div style="height:0.85rem;"></div>', unsafe_allow_html=True)
            render_questionnaire_badge(has_questionnaire)
        with probability_col:
            st.markdown(
                f'<div class="student-probability">{probability:.2f}%</div>',
                unsafe_allow_html=True,
            )
        with notification_col:
            st.markdown('<div style="height:0.85rem;"></div>', unsafe_allow_html=True)
            render_notification_badge(notification_status)
        with action_col:
            st.markdown('<div style="height:0.65rem;"></div>', unsafe_allow_html=True)
            if st.button(
                "View",
                key=f"{key_prefix}_view_{index}_{record_id}",
                use_container_width=True,
            ):
                return_pages = {
                    "csv_results": "Upload CSV",
                    "history_results": "Student List",
                    "timeline_results": "Student Timeline",
                }
                st.session_state.prediction_detail_record = row.to_dict()
                st.session_state.prediction_detail_return = return_pages.get(
                    key_prefix, "Student List"
                )
                st.session_state.page = "prediction_detail"
                st.rerun()

        st.markdown('<div class="student-row-divider"></div>', unsafe_allow_html=True)

    st.info("Press View beside a student to open the full result page.")

    if total_pages > 1:
        prev_col, page_col, next_col = st.columns([1, 2, 1])
        with prev_col:
            if st.button(
                "Previous",
                key=f"{key_prefix}_previous",
                disabled=current_page == 1,
                use_container_width=True,
            ):
                st.session_state[page_key] = current_page - 1
                st.rerun()
        with page_col:
            st.markdown(
                f'<div class="pager-label">Page {current_page} of {total_pages}</div>',
                unsafe_allow_html=True,
            )
        with next_col:
            if st.button(
                "Next",
                key=f"{key_prefix}_next",
                disabled=current_page == total_pages,
                use_container_width=True,
            ):
                st.session_state[page_key] = current_page + 1
                st.rerun()


def prediction_detail_page():
    return_page = st.session_state.get("prediction_detail_return", "Dashboard")
    if return_page == "Prediction History":
        return_page = "Student List"
    title_col, back_col = st.columns([4, 1])
    with title_col:
        st.title("Student Prediction Result")
    with back_col:
        st.markdown('<div style="height:0.8rem;"></div>', unsafe_allow_html=True)
        if st.button(f"Back to {return_page}", use_container_width=True):
            if return_page in {
                "Dashboard",
                "Model Information",
                "Risk Trend",
                "Manual Prediction",
                "Upload CSV",
                "Student List",
                "Student Timeline",
                "Manage Users",
                "Audit Logs",
            }:
                st.session_state.sidebar_selected_page = return_page
            st.session_state.page = "dashboard"
            st.rerun()

    record = st.session_state.get("prediction_detail_record")
    if not record:
        st.warning("No student result selected.")
        return

    render_prediction_detail(record)


def home_page():
    logo_uri = image_data_uri(str(LOGO_FILE))
    if logo_uri:
        logo_html = f'<img src="{logo_uri}" alt="AURA logo">'
    else:
        logo_html = '<div class="home-logo-fallback">AURA</div>'

    st.markdown(
        '<div class="home-hero">'
        f'<div class="home-logo-card">{logo_html}</div>'
        '<div class="home-copy">'
        '<div class="home-kicker">Academic risk intelligence</div>'
        '<h1>See the risk sooner. Act while it matters.</h1>'
        '<p>AURA brings prediction, student context, and advisor action into one calm '
        'workspace—so support teams can move from signal to meaningful intervention.</p>'
        '<div class="home-stat-row">'
        '<div class="home-stat"><strong>Predict</strong><span>Manual and batch risk analysis</span></div>'
        '<div class="home-stat"><strong>Prioritize</strong><span>Clear risk levels and contributing factors</span></div>'
        '<div class="home-stat"><strong>Support</strong><span>Advisor notes, alerts, and student history</span></div>'
        '</div>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="home-action-title">Enter the workspace</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button(
            "Login",
            key="home_login",
            type="primary",
            use_container_width=True,
        ):
            st.session_state.page = "login"
            st.rerun()
    with c2:
        if st.button(
            "Create an account",
            key="home_register",
            use_container_width=True,
        ):
            st.session_state.page = "register"
            st.rerun()

    st.markdown(
        """
        <div style="margin-top: 0.75rem;">
            <a href="mailto:aura.taylorproject.ai@gmail.com"
               style="
                   display: inline-flex;
                   align-items: center;
                   justify-content: center;
                   width: 100%;
                   min-height: 2.7rem;
                   border-radius: 8px;
                   border: 1px solid #22c55e;
                   background: linear-gradient(135deg, #16a34a, #15803d);
                   color: #ffffff;
                   font-weight: 850;
                   text-decoration: none;
                   box-shadow: 0 8px 22px rgba(22, 163, 74, 0.22);
               ">
                📧 Contact Us
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="home-feature-grid">'
        '<div class="home-feature" data-index="01"><h3>Early warning, clearly ranked</h3>'
        '<p>Scan student risk counts and trends without digging through visual noise.</p></div>'
        '<div class="home-feature" data-index="02"><h3>Evidence behind every result</h3>'
        '<p>Review probability, model factors, course context, and recommended next actions.</p></div>'
        '<div class="home-feature" data-index="03"><h3>A connected advisor workflow</h3>'
        '<p>Move between student lists, timelines, notifications, and AI-assisted notes.</p></div>'
        '</div>',
        unsafe_allow_html=True,
    )


def login_page():
    st.markdown('<div class="auth-page-marker"></div>', unsafe_allow_html=True)
    show_logo(120)
    st.title("Login")
    st.write("Enter your email and password. AURA will send a verification code.")
    if st.session_state.login_success_message:
        st.success(st.session_state.login_success_message)
        st.session_state.login_success_message = ""
    email = st.text_input("Email Address")
    password = st.text_input("Password", type="password")

    login_col, forgot_col = st.columns(2)
    with login_col:
        if st.button("Login and Send OTP", use_container_width=True):
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                ok, role, staff_id = check_login(email, password)
                if ok:
                    generate_and_send_otp(email.strip().lower(), role, staff_id)
                else:
                    st.error("Invalid email or password.")

    with forgot_col:
        if st.button("Forgot Password?", use_container_width=True):
            st.session_state.page = "forgot_password"
            st.rerun()

    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()


def forgot_password_page():
    st.markdown('<div class="auth-page-marker"></div>', unsafe_allow_html=True)
    show_logo(120)
    st.title("Forgot Password")
    st.write("Enter your registered email to receive a reset OTP.")
    email = st.text_input("Registered Email Address", key="forgot_email")

    if st.button("Send Reset OTP", use_container_width=True):
        if not email:
            st.error("Please enter your email.")
        else:
            user_row = lookup_user_for_reset_by_email(email)
            if user_row is None:
                st.error("No account was found with that email.")
            else:
                role, staff_id = user_row
                if is_student_role(role):
                    account_ok, account_message = True, ""
                else:
                    account_ok, account_message = validate_institutional_account(
                        email, staff_id, role
                    )
                if not account_ok:
                    st.error(account_message)
                else:
                    generate_and_send_reset_otp(email.strip().lower(), staff_id)

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


def reset_password_page():
    st.markdown('<div class="auth-page-marker"></div>', unsafe_allow_html=True)
    show_logo(120)
    st.title("Reset Password")
    email = st.session_state.pending_reset_email
    staff_id = st.session_state.pending_reset_staff_id
    if not email or not staff_id:
        st.warning("No password reset is waiting for verification.")
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()
        return

    st.write(f"Enter the reset OTP sent to: {email}")
    if st.session_state.reset_email_sent:
        st.success("Reset OTP was sent to your email.")
    elif ALLOW_LOCAL_OTP_FALLBACK:
        st.warning("Email sending failed, but local testing mode is enabled.")
        st.info(f"Testing reset OTP: {st.session_state.reset_otp}")
    else:
        st.error("Reset OTP email was not sent.")
        with st.expander("Show email error reason"):
            st.write(st.session_state.reset_error_message)

    if st.session_state.reset_otp_expiry is not None:
        remaining = st.session_state.reset_otp_expiry - app_now()
        if remaining.total_seconds() > 0:
            minutes = int(remaining.total_seconds() // 60)
            seconds = int(remaining.total_seconds() % 60)
            st.caption(f"Reset OTP expires in {minutes} minutes and {seconds} seconds.")
        else:
            st.warning("Reset OTP expired. Please resend OTP.")

    otp_input = st.text_input("Enter Reset OTP")
    new_password = st.text_input("New Password", type="password")
    confirm_password = st.text_input("Confirm New Password", type="password")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset Password", use_container_width=True):
            if st.session_state.reset_otp_expiry is None:
                st.error("No reset OTP was generated.")
            elif app_now() > st.session_state.reset_otp_expiry:
                st.error("Reset OTP expired. Please resend OTP.")
            elif otp_input.strip() != st.session_state.reset_otp:
                st.error("Wrong reset OTP. Please try again.")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters.")
            elif new_password != confirm_password:
                st.error("Passwords do not match.")
            elif update_user_password(email, staff_id, new_password):
                user_row = lookup_user_for_reset_by_email(email)
                role = user_row[0] if user_row is not None else "advisor"
                st.session_state.pending_reset_email = ""
                st.session_state.pending_reset_staff_id = ""
                st.session_state.reset_otp = ""
                st.session_state.reset_otp_expiry = None
                st.session_state.reset_email_sent = False
                st.session_state.reset_error_message = ""
                st.session_state.logged_in = True
                st.session_state.user_email = email.strip().lower()
                st.session_state.user_role = role
                st.session_state.user_staff_id = normalize_staff_id(staff_id)
                st.session_state.sidebar_selected_page = "Dashboard"
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Could not reset the password for this account.")

    with c2:
        if st.button("Resend Reset OTP", use_container_width=True):
            generate_and_send_reset_otp(email, staff_id)

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


def register_page():
    st.markdown('<div class="auth-page-marker"></div>', unsafe_allow_html=True)
    show_logo(120)
    st.title("Register Account")
    name = st.text_input("Full Name")
    email = st.text_input("Email Address")
    staff_id = st.text_input("ID Number / Student ID")
    password = st.text_input("Password", type="password")

    st.markdown("**Choose Account Type**")
    st.session_state.setdefault("register_account_type", "administrator")
    admin_col, advisor_col, student_col = st.columns(3)
    with admin_col:
        if st.button(
            "Administrator",
            key="register_type_administrator",
            type=(
                "primary"
                if st.session_state.register_account_type == "administrator"
                else "secondary"
            ),
            use_container_width=True,
        ):
            st.session_state.register_account_type = "administrator"
            st.rerun()
    with advisor_col:
        if st.button(
            "Advisor",
            key="register_type_advisor",
            type=(
                "primary"
                if st.session_state.register_account_type == "advisor"
                else "secondary"
            ),
            use_container_width=True,
        ):
            st.session_state.register_account_type = "advisor"
            st.rerun()
    with student_col:
        if st.button(
            "Student",
            key="register_type_student",
            type=(
                "primary"
                if st.session_state.register_account_type == "student"
                else "secondary"
            ),
            use_container_width=True,
        ):
            st.session_state.register_account_type = "student"
            st.rerun()

    selected_role_label = role_display_name(st.session_state.register_account_type)
    st.caption(f"Selected account type: {selected_role_label}")

    if st.button("Create Account", use_container_width=True):
        role = st.session_state.register_account_type
        if role == "student":
            account_ok = is_valid_email(email)
            account_message = "" if account_ok else "Please enter a valid email address."
            staff_id = normalize_student_id(staff_id)
        else:
            account_ok, account_message = validate_institutional_account(email, staff_id, role)
        if not name or not email or not staff_id or not password:
            st.error("Please fill all fields.")
        elif not account_ok:
            st.error(account_message)
        elif len(password) < 6:
            st.error("Password must be at least 6 characters.")
        else:
            if role != "student":
                csv_ok, csv_message = refresh_authorized_users_from_csv()
                if not csv_ok:
                    st.error(csv_message)
                    return
                if lookup_authorized_user(email, staff_id) is None:
                    st.error(explain_authorized_registration_error(email, staff_id))
                    return
                if role == "administrator" and email.strip().lower() == ADMIN_EMAIL:
                    role = "admin"
            create_status = create_user(name, email, password, role, staff_id)
            if create_status == "created":
                st.success("Account created. Sending OTP.")
                generate_and_send_otp(email.strip().lower(), role, staff_id)
            elif create_status == "already_registered":
                st.error("This account is already registered. Please log in or use Forgot Password.")
            elif create_status == "email_conflict":
                st.error("This email is already registered with a different ID number.")
            elif create_status == "staff_id_conflict":
                st.error("This ID number is already registered with a different email.")
            else:
                st.error("This email or ID number is already registered.")

    if st.button("Back"):
        st.session_state.page = "home"
        st.rerun()


def otp_page():
    st.markdown('<div class="auth-page-marker"></div>', unsafe_allow_html=True)
    show_logo(120)
    st.title("Email Verification")

    email = st.session_state.pending_auth_email
    if not email:
        st.warning("No login is waiting for verification.")
        if st.button("Back to Login"):
            st.session_state.page = "login"
            st.rerun()
        return

    st.write(f"Enter the OTP sent to: {email}")

    if st.session_state.otp_email_sent:
        st.success("OTP was sent to your email.")
    elif ALLOW_LOCAL_OTP_FALLBACK:
        st.warning("Email sending failed, but local testing mode is enabled.")
        st.info(f"Testing OTP: {st.session_state.otp}")
    else:
        st.error("OTP email was not sent.")
        with st.expander("Show email error reason"):
            st.write(st.session_state.otp_error_message)

    if st.session_state.otp_expiry is not None:
        remaining = st.session_state.otp_expiry - app_now()
        if remaining.total_seconds() > 0:
            minutes = int(remaining.total_seconds() // 60)
            seconds = int(remaining.total_seconds() % 60)
            st.caption(f"OTP expires in {minutes} minutes and {seconds} seconds.")
        else:
            st.warning("OTP expired. Please resend OTP.")

    otp_input = st.text_input("Enter OTP")
    c1, c2 = st.columns(2)

    with c1:
        if st.button("Verify OTP", use_container_width=True):
            if st.session_state.otp_expiry is None:
                st.error("No OTP was generated.")
            elif app_now() > st.session_state.otp_expiry:
                st.error("OTP expired. Please resend OTP.")
            elif otp_input.strip() == st.session_state.otp:
                st.session_state.logged_in = True
                st.session_state.user_email = st.session_state.pending_auth_email
                st.session_state.user_role = st.session_state.pending_auth_role
                st.session_state.user_staff_id = st.session_state.pending_auth_staff_id
                st.session_state.pending_auth_email = ""
                st.session_state.pending_auth_role = ""
                st.session_state.pending_auth_staff_id = ""
                st.session_state.otp = ""
                st.session_state.otp_expiry = None
                st.session_state.otp_email_sent = False
                st.session_state.otp_error_message = ""
                st.session_state.otp_last_sent_at = None
                st.session_state.otp_last_sent_email = ""
                st.session_state.otp_send_in_progress = False
                st.session_state.page = "dashboard"
                st.rerun()
            else:
                st.error("Wrong OTP. Please try again.")

    with c2:
        if st.button("Resend OTP", use_container_width=True):
            generate_and_send_otp(
                st.session_state.pending_auth_email,
                st.session_state.pending_auth_role,
                st.session_state.pending_auth_staff_id,
                force_new=True,
            )

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()


def student_questionnaire_page():
    st.title("Student Support Questionnaire")
    student_email = st.session_state.user_email
    student_id = normalize_student_id(st.session_state.user_staff_id)
    st.session_state.setdefault("student_questionnaire_step", 1)
    st.session_state.setdefault("student_course_count", 3)

    st.markdown(
        f"""
        <div class="detail-hero">
            <div>
                <div class="detail-eyebrow">Student Page</div>
                <h2>{safe_html(student_id or "Student ID")}</h2>
                <div class="detail-subtitle">
                    Signed in as: {safe_html(student_email)}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    latest_questionnaire, latest_courses = get_latest_questionnaire(student_id)
    if latest_questionnaire:
        st.success(
            "Questionnaire completed. Last updated: "
            f"{clean_display_value(latest_questionnaire.get('submitted_at'))} "
            f"({clean_display_value(latest_questionnaire.get('semester'), 'Semester not set')})"
        )
    else:
        st.info("Please complete the questionnaire so advisors can personalize support.")

    def empty_radio(label, options, key):
        return st.radio(label, options, index=None, key=key)

    def current_courses_from_widgets():
        courses = []
        for course_index in range(int(st.session_state.student_course_count)):
            course_name = st.session_state.get(f"student_course_name_{course_index}", "")
            credits = st.session_state.get(f"student_course_credits_{course_index}", 3.0)
            hours = st.session_state.get(f"student_course_hours_{course_index}", 0.0)
            if str(course_name).strip():
                courses.append(
                    {
                        "course_name": str(course_name).strip(),
                        "course_credits": credits,
                        "weekly_study_hours": hours,
                    }
                )
        return courses

    with st.container(border=True):
        st.subheader("Section 1: Current Semester Courses")
        st.selectbox(
            "Q1. Which academic semester does this questionnaire belong to?",
            SEMESTER_OPTIONS,
            key="student_questionnaire_semester",
            help=(
                "Select the student's academic progression semester. Short semesters "
                "can be recorded as .5, for example Semester 1.5 or Semester 2.5."
            ),
        )
        st.caption("Recommended weekly study time is calculated as course credits x 2.")
        count_col, remove_col, add_col = st.columns([5, 1, 1])
        with count_col:
            st.markdown(
                f"**Q2. How many courses do you want to enter?**  \n"
                f"Current courses: **{st.session_state.student_course_count}**"
            )
        with remove_col:
            if st.button("-", key="onepage_course_remove", use_container_width=True, disabled=st.session_state.student_course_count <= 1):
                st.session_state.student_course_count -= 1
                st.rerun()
        with add_col:
            if st.button("+", key="onepage_course_add", use_container_width=True, disabled=st.session_state.student_course_count >= 8):
                st.session_state.student_course_count += 1
                st.rerun()

        for index in range(int(st.session_state.student_course_count)):
            st.session_state.setdefault(f"student_course_credits_{index}", 3.0)
            st.session_state.setdefault(f"student_course_hours_{index}", 0.0)
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.text_input(
                    f"Q2.{index + 1} Course {index + 1} Name",
                    key=f"student_course_name_{index}",
                    placeholder="Example: Machine Learning",
                )
            with c2:
                st.number_input(
                    f"Credits {index + 1}",
                    min_value=1.0,
                    max_value=8.0,
                    step=0.5,
                    key=f"student_course_credits_{index}",
                )
            with c3:
                st.number_input(
                    f"Hours/week {index + 1}",
                    min_value=0.0,
                    max_value=80.0,
                    step=0.5,
                    key=f"student_course_hours_{index}",
                )

    with st.container(border=True):
        st.subheader("Section 2: Academic Progress")
        empty_radio(
            "Q2. What is your current attendance rate?",
            ["Below 50%", "50%-59%", "60%-69%", "70%-79%", "80%-89%", "90% and above"],
            "student_attendance_rate",
        )
        empty_radio(
            "Q3. What is your current assignment submission status?",
            ["Always on time", "Usually on time", "Sometimes late", "Often late", "Always late"],
            "student_assignment_status",
        )
        empty_radio(
            "Q4. What is your average coursework mark range?",
            ["Below 40", "40-49", "50-59", "60-69", "70-79", "80 and above"],
            "student_coursework_mark",
        )
        empty_radio(
            "Q5. Would you like AURA to track your study progress automatically?",
            ["Yes", "No"],
            "student_auto_tracking",
        )
        st.multiselect(
            "Q6. Which performance indicators should AURA monitor for you?",
            ["Study hours", "Attendance", "Assignment submission status", "Coursework marks", "GPA"],
            key="student_performance_indicators",
        )
        empty_radio(
            "Q7. How often would you like performance updates?",
            ["Daily", "Weekly", "Monthly", "Before deadlines only", "Only when a warning is detected"],
            "student_update_frequency",
        )

    with st.container(border=True):
        st.subheader("Section 3: Alerts and Suggestions")
        st.multiselect(
            "Q8. What type of alerts would be most useful for you?",
            [
                "Low attendance warning",
                "Assignment deadline reminder",
                "Low coursework mark warning",
                "Low study hour reminder",
            ],
            key="student_alert_types",
        )
        st.multiselect(
            "Q9. What kind of suggestions would help you improve your performance?",
            [
                "Study plan recommendations",
                "Time management tips",
                "Assignment planning tips",
                "Revision reminders",
                "Consultation with lecturer/mentor",
                "GPA improvement suggestions",
            ],
            key="student_suggestions_needed",
        )
        st.multiselect(
            "Q10. How would you like AURA to send alerts?",
            ["In-app notification", "Email", "WhatsApp/Telegram", "Dashboard only", "All of the above"],
            key="student_alert_method",
        )

    with st.container(border=True):
        st.subheader("Section 4: Permission")
        empty_radio(
            "Q11. Would you allow AURA to flag you as at risk when your study behavior shows warning signs?",
            ["Yes", "No"],
            "student_allow_risk_flagging",
        )
        empty_radio(
            "Q12. Would you like to be reminded if AURA detects that you are at high risk?",
            ["Yes", "No"],
            "student_high_risk_reminder_consent",
        )

    with st.container(border=True):
        st.subheader("Section 5: Support and Communication")
        st.multiselect(
            "Q13. If AURA detects a risk, what support would you prefer first?",
            [
                "Study reminder",
                "Academic counselling",
                "Lecturer notification",
                "Time management advice",
                "Performance improvement tips",
            ],
            key="student_preferred_support",
        )
        st.multiselect(
            "Q14. How do you usually communicate with the administration when you need academic help?",
            ["Email", "Student portal", "In person", "Phone call", "WhatsApp / Telegram", "I rarely contact the administration"],
            key="student_communication_method",
        )
        empty_radio(
            "Q15. How easy is it for you to communicate with the administration?",
            [1, 2, 3, 4, 5],
            "student_communication_ease",
        )
        st.multiselect(
            "Q16. What is the main barrier when contacting the administration?",
            [
                "Slow response",
                "No clear contact channel",
                "Difficult to get an appointment",
                "Unclear procedures",
                "I do not know who to contact",
                "No barrier",
            ],
            key="student_main_barrier",
        )
        st.multiselect(
            "Q17. What kind of support would you like from the administration?",
            [
                "Academic guidance",
                "Deadline clarification",
                "Performance feedback",
                "Risk warning / early alert",
                "Personal consultation",
                "Study planning support",
            ],
            key="student_administration_support",
        )
        st.multiselect(
            "Q18. What is your preferred way for AURA to contact you?",
            ["Email", "WhatsApp", "Telegram", "In-app notification", "Student Portal Notification"],
            key="student_preferred_contact_method",
        )
        st.text_area(
            "Q19. Please provide your preferred contact details.",
            placeholder="Example: WhatsApp number, Telegram username, or preferred email",
            key="student_preferred_contact_details",
        )

    if st.button("Submit Questionnaire", key="onepage_submit_questionnaire", use_container_width=True, type="primary"):
        courses = current_courses_from_widgets()
        if not student_id:
            st.error("Your student ID is missing. Please register again with a valid student ID.")
        elif not courses:
            st.error("Please enter at least one course name.")
        else:
            answers = {
                "semester": st.session_state.get("student_questionnaire_semester") or "Semester 1",
                "attendance_rate": st.session_state.get("student_attendance_rate") or "",
                "assignment_status": st.session_state.get("student_assignment_status") or "",
                "coursework_mark": st.session_state.get("student_coursework_mark") or "",
                "auto_tracking": st.session_state.get("student_auto_tracking") or "",
                "performance_indicators": st.session_state.get("student_performance_indicators", []),
                "update_frequency": st.session_state.get("student_update_frequency") or "",
                "alert_types": st.session_state.get("student_alert_types", []),
                "suggestions_needed": st.session_state.get("student_suggestions_needed", []),
                "alert_method": st.session_state.get("student_alert_method", []),
                "badge_motivation": 0,
                "badge_achievements": [],
                "reward_encouragement": 0,
                "allow_risk_flagging": st.session_state.get("student_allow_risk_flagging") or "",
                "high_risk_reminder_consent": st.session_state.get("student_high_risk_reminder_consent") or "",
                "preferred_support": st.session_state.get("student_preferred_support", []),
                "communication_method": st.session_state.get("student_communication_method", []),
                "communication_ease": st.session_state.get("student_communication_ease") or 0,
                "main_barrier": st.session_state.get("student_main_barrier", []),
                "administration_support": st.session_state.get("student_administration_support", []),
                "preferred_contact_method": st.session_state.get("student_preferred_contact_method", []),
                "preferred_contact_details": st.session_state.get("student_preferred_contact_details", ""),
            }
            save_student_questionnaire(student_email, student_id, answers, courses)
            st.success("Questionnaire submitted successfully.")
            st.rerun()

    if latest_questionnaire and not latest_courses.empty:
        st.subheader("Latest Course Study Load")
        st.dataframe(latest_courses, use_container_width=True)

    return

    section_titles = {
        1: "Courses",
        2: "Academic Progress",
        3: "Alerts and Suggestions",
        4: "Permission",
        5: "Support and Communication",
    }
    step = int(st.session_state.student_questionnaire_step)
    st.progress(step / 5)
    st.caption(f"Section {step} of 5: {section_titles[step]}")

    def save_course_widgets():
        courses_data = []
        for course_index in range(int(st.session_state.student_course_count)):
            courses_data.append(
                {
                    "course_name": st.session_state.get(
                        f"student_course_name_{course_index}", ""
                    ),
                    "course_credits": st.session_state.get(
                        f"student_course_credits_{course_index}", 3.0
                    ),
                    "weekly_study_hours": st.session_state.get(
                        f"student_course_hours_{course_index}", 0.0
                    ),
                }
            )
        st.session_state.student_courses_data = courses_data

    def restore_course_widget_defaults(index):
        courses_data = st.session_state.get("student_courses_data", [])
        if index >= len(courses_data):
            return "", 3.0, 0.0
        course = courses_data[index]
        return (
            str(course.get("course_name", "")),
            float(course.get("course_credits", 3.0)),
            float(course.get("weekly_study_hours", 0.0)),
        )

    def save_questionnaire_widget_values():
        field_keys = [
            "student_attendance_rate",
            "student_assignment_status",
            "student_coursework_mark",
            "student_auto_tracking",
            "student_performance_indicators",
            "student_update_frequency",
            "student_alert_types",
            "student_suggestions_needed",
            "student_alert_method",
            "student_allow_risk_flagging",
            "student_high_risk_reminder_consent",
            "student_preferred_support",
            "student_communication_method",
            "student_communication_ease",
            "student_main_barrier",
            "student_administration_support",
            "student_preferred_contact_method",
            "student_preferred_contact_details",
        ]
        for field_key in field_keys:
            widget_key = f"{field_key}_widget"
            if widget_key in st.session_state:
                st.session_state[field_key] = st.session_state[widget_key]

    def save_current_section():
        save_course_widgets()
        save_questionnaire_widget_values()

    def go_to_step(next_step):
        save_current_section()
        st.session_state.student_questionnaire_step = max(1, min(5, next_step))
        st.rerun()

    def collect_courses():
        save_course_widgets()
        collected_courses = []
        for course in st.session_state.get("student_courses_data", []):
            course_name = course.get("course_name", "")
            credits = course.get("course_credits", 3.0)
            hours = course.get("weekly_study_hours", 0.0)
            if str(course_name).strip():
                collected_courses.append(
                    {
                        "course_name": str(course_name).strip(),
                        "course_credits": credits,
                        "weekly_study_hours": hours,
                    }
                )
        return collected_courses

    def persistent_radio(label, options, state_key):
        current_value = st.session_state.get(state_key)
        index = options.index(current_value) if current_value in options else None
        selected = st.radio(label, options, index=index, key=f"{state_key}_widget")
        if selected is not None:
            st.session_state[state_key] = selected
        return st.session_state.get(state_key, "")

    def persistent_multiselect(label, options, state_key):
        current_values = st.session_state.get(state_key, [])
        selected = st.multiselect(
            label,
            options,
            default=[value for value in current_values if value in options],
            key=f"{state_key}_widget",
        )
        st.session_state[state_key] = selected
        return selected

    def persistent_text_area(label, state_key, placeholder=""):
        current_value = st.session_state.get(state_key, "")
        value = st.text_area(
            label,
            value=current_value,
            placeholder=placeholder,
            key=f"{state_key}_widget",
        )
        st.session_state[state_key] = value
        return value

    section_options = [
        f"Section {number}: {title}" for number, title in section_titles.items()
    ]
    selected_section = st.radio(
        "Go to section",
        section_options,
        index=step - 1,
        horizontal=True,
        key=f"student_section_picker_{step}",
    )
    selected_step = section_options.index(selected_section) + 1
    if selected_step != step:
        go_to_step(selected_step)

    if step == 1:
        st.subheader("Section 1: Current Semester Courses")
        st.caption("Recommended weekly study time is calculated as course credits x 2.")
        count_col, remove_col, add_col = st.columns([5, 1, 1])
        with count_col:
            st.markdown(
                f"**Q1. How many courses do you want to enter?**  \n"
                f"Current courses: **{st.session_state.student_course_count}**"
            )
        with remove_col:
            if st.button("-", use_container_width=True, disabled=st.session_state.student_course_count <= 1):
                save_course_widgets()
                st.session_state.student_course_count -= 1
                st.session_state.student_courses_data = st.session_state.student_courses_data[
                    : st.session_state.student_course_count
                ]
                st.rerun()
        with add_col:
            if st.button("+", use_container_width=True, disabled=st.session_state.student_course_count >= 8):
                save_course_widgets()
                st.session_state.student_course_count += 1
                st.rerun()

        for index in range(int(st.session_state.student_course_count)):
            course_name_value, credits_value, hours_value = restore_course_widget_defaults(index)
            c1, c2, c3 = st.columns([3, 1, 1])
            with c1:
                st.text_input(
                    f"Q1.{index + 1} Course {index + 1} Name",
                    value=course_name_value,
                    key=f"student_course_name_{index}",
                    placeholder="Example: Machine Learning",
                )
            with c2:
                st.number_input(
                    f"Credits {index + 1}",
                    min_value=1.0,
                    max_value=8.0,
                    value=credits_value,
                    step=0.5,
                    key=f"student_course_credits_{index}",
                )
            with c3:
                st.number_input(
                    f"Hours/week {index + 1}",
                    min_value=0.0,
                    max_value=80.0,
                    value=hours_value,
                    step=0.5,
                    key=f"student_course_hours_{index}",
                )

        if st.button("Next Section", use_container_width=True):
            if not collect_courses():
                st.error("Please enter at least one course name.")
            else:
                go_to_step(2)

    elif step == 2:
        st.subheader("Section 2: Academic Progress")
        persistent_radio(
            "Q2. What is your current attendance rate?",
            ["Below 50%", "50%-59%", "60%-69%", "70%-79%", "80%-89%", "90% and above"],
            "student_attendance_rate",
        )
        persistent_radio(
            "Q3. What is your current assignment submission status?",
            ["Always on time", "Usually on time", "Sometimes late", "Often late", "Always late"],
            "student_assignment_status",
        )
        persistent_radio(
            "Q4. What is your average coursework mark range?",
            ["Below 40", "40-49", "50-59", "60-69", "70-79", "80 and above"],
            "student_coursework_mark",
        )
        persistent_radio(
            "Q5. Would you like AURA to track your study progress automatically?",
            ["Yes", "No"],
            "student_auto_tracking",
        )
        persistent_multiselect(
            "Q6. Which performance indicators should AURA monitor for you?",
            ["Study hours", "Attendance", "Assignment submission status", "Coursework marks", "GPA"],
            "student_performance_indicators",
        )
        persistent_radio(
            "Q7. How often would you like performance updates?",
            ["Daily", "Weekly", "Monthly", "Before deadlines only", "Only when a warning is detected"],
            "student_update_frequency",
        )
        back_col, next_col = st.columns(2)
        with back_col:
            if st.button("Back", use_container_width=True):
                go_to_step(1)
        with next_col:
            if st.button("Next Section", use_container_width=True):
                go_to_step(3)

    elif step == 3:
        st.subheader("Section 3: Alerts and Suggestions")
        persistent_multiselect(
            "Q8. What type of alerts would be most useful for you?",
            [
                "Low attendance warning",
                "Assignment deadline reminder",
                "Low coursework mark warning",
                "Low study hour reminder",
            ],
            "student_alert_types",
        )
        persistent_multiselect(
            "Q9. What kind of suggestions would help you improve your performance?",
            [
                "Study plan recommendations",
                "Time management tips",
                "Assignment planning tips",
                "Revision reminders",
                "Consultation with lecturer/mentor",
                "GPA improvement suggestions",
            ],
            "student_suggestions_needed",
        )
        persistent_multiselect(
            "Q10. How would you like AURA to send alerts?",
            ["In-app notification", "Email", "WhatsApp/Telegram", "Dashboard only", "All of the above"],
            "student_alert_method",
        )
        back_col, next_col = st.columns(2)
        with back_col:
            if st.button("Back", use_container_width=True):
                go_to_step(2)
        with next_col:
            if st.button("Next Section", use_container_width=True):
                go_to_step(4)

    elif step == 4:
        st.subheader("Section 4: Permission")
        persistent_radio(
            "Q11. Would you allow AURA to flag you as at risk when your study behavior shows warning signs?",
            ["Yes", "No"],
            "student_allow_risk_flagging",
        )
        persistent_radio(
            "Q12. Would you like to be reminded if AURA detects that you are at high risk?",
            ["Yes", "No"],
            "student_high_risk_reminder_consent",
        )
        back_col, next_col = st.columns(2)
        with back_col:
            if st.button("Back", use_container_width=True):
                go_to_step(3)
        with next_col:
            if st.button("Next Section", use_container_width=True):
                go_to_step(5)

    elif step == 5:
        st.subheader("Section 5: Support and Communication")
        persistent_multiselect(
            "Q13. If AURA detects a risk, what support would you prefer first?",
            [
                "Study reminder",
                "Academic counselling",
                "Lecturer notification",
                "Time management advice",
                "Performance improvement tips",
            ],
            "student_preferred_support",
        )
        persistent_multiselect(
            "Q14. How do you usually communicate with the administration when you need academic help?",
            ["Email", "Student portal", "In person", "Phone call", "WhatsApp / Telegram", "I rarely contact the administration"],
            "student_communication_method",
        )
        persistent_radio(
            "Q15. How easy is it for you to communicate with the administration?",
            [1, 2, 3, 4, 5],
            "student_communication_ease",
        )
        persistent_multiselect(
            "Q16. What is the main barrier when contacting the administration?",
            [
                "Slow response",
                "No clear contact channel",
                "Difficult to get an appointment",
                "Unclear procedures",
                "I do not know who to contact",
                "No barrier",
            ],
            "student_main_barrier",
        )
        persistent_multiselect(
            "Q17. What kind of support would you like from the administration?",
            [
                "Academic guidance",
                "Deadline clarification",
                "Performance feedback",
                "Risk warning / early alert",
                "Personal consultation",
                "Study planning support",
            ],
            "student_administration_support",
        )
        persistent_multiselect(
            "Q18. What is your preferred way for AURA to contact you?",
            ["Email", "WhatsApp", "Telegram", "In-app notification", "Student Portal Notification"],
            "student_preferred_contact_method",
        )
        persistent_text_area(
            "Q19. Please provide your preferred contact details.",
            "student_preferred_contact_details",
            placeholder="Example: WhatsApp number, Telegram username, or preferred email",
        )

        back_col, submit_col = st.columns(2)
        with back_col:
            if st.button("Back", use_container_width=True):
                go_to_step(4)
        with submit_col:
            submitted = st.button("Submit Questionnaire", use_container_width=True, type="primary")
    else:
        submitted = False

    if step != 5:
        submitted = False

    if submitted:
        courses = collect_courses()
        if not student_id:
            st.error("Your student ID is missing. Please register again with a valid student ID.")
        elif not courses:
            st.error("Please enter at least one course name.")
        else:
            answers = {
                "attendance_rate": st.session_state.get("student_attendance_rate", ""),
                "assignment_status": st.session_state.get("student_assignment_status", ""),
                "coursework_mark": st.session_state.get("student_coursework_mark", ""),
                "auto_tracking": st.session_state.get("student_auto_tracking", ""),
                "performance_indicators": st.session_state.get("student_performance_indicators", []),
                "update_frequency": st.session_state.get("student_update_frequency", ""),
                "alert_types": st.session_state.get("student_alert_types", []),
                "suggestions_needed": st.session_state.get("student_suggestions_needed", []),
                "alert_method": st.session_state.get("student_alert_method", []),
                "badge_motivation": 0,
                "badge_achievements": [],
                "reward_encouragement": 0,
                "allow_risk_flagging": st.session_state.get("student_allow_risk_flagging", ""),
                "high_risk_reminder_consent": st.session_state.get("student_high_risk_reminder_consent", ""),
                "preferred_support": st.session_state.get("student_preferred_support", []),
                "communication_method": st.session_state.get("student_communication_method", []),
                "communication_ease": st.session_state.get("student_communication_ease", 3),
                "main_barrier": st.session_state.get("student_main_barrier", []),
                "administration_support": st.session_state.get("student_administration_support", []),
                "preferred_contact_method": st.session_state.get("student_preferred_contact_method", []),
                "preferred_contact_details": st.session_state.get("student_preferred_contact_details", ""),
            }
            save_student_questionnaire(student_email, student_id, answers, courses)
            st.session_state.student_questionnaire_step = 1
            st.success("Questionnaire submitted successfully.")
            st.rerun()

    if latest_questionnaire and not latest_courses.empty:
        st.subheader("Latest Course Study Load")
        st.dataframe(latest_courses, use_container_width=True)


def sidebar_menu():
    with st.sidebar:
        show_sidebar_logo()
        st.markdown(
            f"""
            <div class="sidebar-user-card">
                <div class="sidebar-user-label">Signed in as</div>
                <div class="sidebar-user-value">{safe_html(st.session_state.user_email)}</div>
                <div class="sidebar-user-label">Role</div>
                <div class="sidebar-user-value">{safe_html(st.session_state.user_role)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if is_student_role(st.session_state.user_role):
            main_pages = [
                "Support Questionnaire",
            ]
        elif is_administrator_user():
            main_pages = [
                "Dashboard",
                "Manual Prediction",
                "Upload CSV",
                "Course Results Upload",
                "Model Information",
            ]
        else:
            main_pages = [
                "Dashboard",
                "Student List",
                "Student Timeline",
                "Risk Trend",
                "Model Information",
            ]
        admin_pages = []
        if is_administrator_user():
            admin_pages = ["Manage Users", "Audit Logs"]

        pages = main_pages + admin_pages
        nav_labels = {
            "Support Questionnaire": "Support profile",
            "Dashboard": "Overview",
            "Manual Prediction": "Manual prediction",
            "Upload CSV": "Batch analysis",
            "Course Results Upload": "Course results",
            "Model Information": "Model intelligence",
            "Student List": "Student list",
            "Student Timeline": "Student timeline",
            "Risk Trend": "Risk trend",
            "Manage Users": "Access control",
            "Audit Logs": "Audit trail",
        }

        selected_page = st.session_state.get("sidebar_selected_page", "Dashboard")
        if selected_page not in pages:
            selected_page = pages[0]
            st.session_state.sidebar_selected_page = selected_page

        for page in main_pages:
            if st.button(
                nav_labels.get(page, page),
                key=f"nav_{re.sub(r'[^A-Za-z0-9]+', '_', page).lower()}",
                type="primary" if page == selected_page else "secondary",
                use_container_width=True,
            ):
                st.session_state.sidebar_selected_page = page
                st.rerun()

        if admin_pages:
            st.markdown(
                '<div class="sidebar-section-title">Administration</div>',
                unsafe_allow_html=True,
            )
            for page in admin_pages:
                if st.button(
                    nav_labels.get(page, page),
                    key=f"nav_{re.sub(r'[^A-Za-z0-9]+', '_', page).lower()}",
                    type="primary" if page == selected_page else "secondary",
                    use_container_width=True,
                ):
                    st.session_state.sidebar_selected_page = page
                    st.rerun()

        if st.button("Sign out"):
            st.session_state.clear()
            init_session()
            st.rerun()
    return st.session_state.get("sidebar_selected_page", "Dashboard")


def render_risk_dashboard_overview():
    df = load_history()
    if df.empty:
        st.info("No prediction records yet.")
        return

    risk_order = ["High Risk", "Medium Risk", "Low Risk"]
    risk_colors = {
        "High Risk": "#b4232f",
        "Medium Risk": "#b36b00",
        "Low Risk": "#208454",
    }
    df["predicted_risk"] = df["predicted_risk"].apply(dashboard_risk_label)
    df = df[df["predicted_risk"].isin(risk_order)]
    risk_counts = df["predicted_risk"].value_counts()
    risk_card_classes = {
        "High Risk": "risk-summary-high",
        "Medium Risk": "risk-summary-medium",
        "Low Risk": "risk-summary-low",
    }
    risk_card_captions = {
        "High Risk": "Needs prompt advisor attention",
        "Medium Risk": "Monitor and provide support",
        "Low Risk": "Currently stable",
    }
    risk_cards = []
    for label in risk_order:
        risk_cards.append(
            f'<div class="risk-summary-card {risk_card_classes[label]}">'
            f'<div class="risk-summary-label">{safe_html(label)}</div>'
            f'<div class="risk-summary-value">{int(risk_counts.get(label, 0))}</div>'
            f'<div class="risk-summary-caption">{risk_card_captions[label]}</div>'
            "</div>"
        )
    st.markdown(
        '<div class="risk-summary-grid">' + "".join(risk_cards) + "</div>",
        unsafe_allow_html=True,
    )

    chart_df = pd.DataFrame(
        {
            "Risk Level": risk_order,
            "Count": [int(risk_counts.get(label, 0)) for label in risk_order],
        }
    )
    fig = px.bar(
        chart_df,
        x="Risk Level",
        y="Count",
        color="Risk Level",
        category_orders={"Risk Level": risk_order},
        color_discrete_map=risk_colors,
    )
    fig.update_layout(
        xaxis_title="Risk Level",
        yaxis_title="Count",
        legend_title_text="Risk Level",
    )
    aura_plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Predictions")
    recent_df = df[
        [
            "prediction_time",
            "student_name",
            "student_id",
            "predicted_risk",
            "probability_score",
            "semester",
            "gpa",
            "stress_index",
        ]
    ].head(20).copy()
    st.dataframe(
        recent_df.style.apply(dashboard_recent_row_style, axis=1),
        use_container_width=True,
    )


def render_course_failure_analysis():
    st.subheader("Course Failure Analysis")
    st.caption(
        "Monitor failed student counts by course. Final and Resit uploads are treated as the same semester; Resit updates the current result."
    )

    with get_conn() as conn:
        current_df = course_failures.load_course_results_current(conn)

    if current_df.empty:
        st.info("No course result records yet. Admin can upload course_results.csv from Course Results Upload.")
        return

    current_df["academic_year"] = pd.to_numeric(
        current_df["academic_year"], errors="coerce"
    )
    current_df = current_df.dropna(subset=["academic_year"]).copy()
    current_df["academic_year"] = current_df["academic_year"].astype(int)
    if current_df.empty:
        st.info("No valid academic year values are available in course results.")
        return
    latest_year, latest_term = course_failures.latest_year_semester(current_df)

    years = sorted(current_df["academic_year"].dropna().astype(int).unique(), reverse=True)
    terms = ["February", "April", "September"]
    default_year_index = years.index(latest_year) if latest_year in years else 0
    default_term_index = terms.index(latest_term) if latest_term in terms else 0

    filter_cols = st.columns([1, 1, 1.4, 2])
    with filter_cols[0]:
        selected_year = st.selectbox(
            "Academic Year",
            years,
            index=default_year_index,
            key="course_failure_year",
        )
    with filter_cols[1]:
        selected_term = st.selectbox(
            "Semester Term",
            terms,
            index=default_term_index,
            key="course_failure_term",
        )
    with filter_cols[2]:
        ranking_type = st.selectbox(
            "Ranking Type",
            [
                "Selected Semester Ranking",
                "Overall Cumulative Ranking",
                "Specific Course Search",
            ],
            key="course_failure_ranking_type",
        )
    with filter_cols[3]:
        course_search = st.text_input(
            "Course Search",
            placeholder="Example: Machine Learning or CRS001",
            key="course_failure_course_search",
        )

    semester_df = current_df[
        (current_df["academic_year"].eq(int(selected_year)))
        & (current_df["semester_term"].astype(str).eq(selected_term))
    ].copy()

    summary_base = semester_df
    if ranking_type == "Overall Cumulative Ranking":
        summary_base = current_df
    elif ranking_type == "Specific Course Search" and course_search.strip():
        search = course_search.strip()
        summary_base = current_df[
            current_df["course_name"].astype(str).str.contains(
                search, case=False, na=False, regex=False
            )
            | current_df["course_code"].astype(str).str.contains(
                search, case=False, na=False, regex=False
            )
        ].copy()

    summary_df = course_failures.course_failure_summary(summary_base)
    total_failed = int(summary_df["fail_number"].sum()) if not summary_df.empty else 0
    failed_courses_df = summary_df[summary_df["fail_number"] > 0]
    highest_course = (
        failed_courses_df.iloc[0]["course_name"] if not failed_courses_df.empty else "No failures"
    )
    courses_with_failures = int((summary_df["fail_number"] > 0).sum()) if not summary_df.empty else 0
    average_fail_rate = float(summary_df["fail_rate"].mean()) if not summary_df.empty else 0.0

    card_cols = st.columns(4)
    card_cols[0].metric("Total Failed Students", total_failed)
    card_cols[1].metric("Highest Failure Course", highest_course)
    card_cols[2].metric("Courses with Failures", courses_with_failures)
    card_cols[3].metric("Average Fail Rate", f"{average_fail_rate:.1f}%")

    if latest_year and latest_term:
        latest_df = current_df[
            (current_df["academic_year"].eq(int(latest_year)))
            & (current_df["semester_term"].astype(str).eq(latest_term))
        ]
        latest_top = course_failures.course_failure_summary(latest_df).head(3)
        if not latest_top.empty:
            st.markdown(f"### Latest Semester Top 3: {latest_year} {latest_term}")
            st.dataframe(
                latest_top[
                    [
                        "course_code",
                        "course_name",
                        "total_students",
                        "fail_number",
                        "fail_rate",
                    ]
                ].style.format({"fail_rate": "{:.1f}%"}),
                use_container_width=True,
                hide_index=True,
            )

    if ranking_type == "Specific Course Search":
        if not course_search.strip():
            st.info("Enter a course name or course code to view its failure trend.")
            return
        trend_df = course_failures.course_trend_summary(current_df, course_search)
        if trend_df.empty:
            st.info("No course result records match this search.")
            return
        st.markdown("### Specific Course Failure Trend")
        trend_display = trend_df.rename(
            columns={
                "academic_year": "Academic Year",
                "semester_term": "Semester",
                "total_students": "Total Students",
                "fail_number": "Fail Number",
                "fail_rate": "Fail Rate",
            }
        )
        st.dataframe(
            trend_display.style.format({"Fail Rate": "{:.1f}%"}),
            use_container_width=True,
            hide_index=True,
        )
        fig = px.bar(
            trend_df,
            x=trend_df["academic_year"].astype(str) + " " + trend_df["semester_term"],
            y="fail_number",
            text="fail_number",
            labels={"x": "Semester", "fail_number": "Fail Number"},
        )
        fig.update_layout(xaxis_title="Semester", yaxis_title="Fail Number")
        aura_plotly_chart(fig, use_container_width=True)
        return

    if summary_df.empty:
        st.info("No course failure data matches the selected filters.")
        return

    chart_df = summary_df[summary_df["fail_number"] > 0].head(15).copy()
    if chart_df.empty:
        st.success("No failed students found for this view.")
    else:
        chart_title = (
            f"Fail Number by Course: {selected_year} {selected_term}"
            if ranking_type == "Selected Semester Ranking"
            else "Cumulative Fail Number by Course"
        )
        st.markdown(f"### {chart_title}")
        fig = px.bar(
            chart_df,
            x="course_name",
            y="fail_number",
            color="fail_rate",
            text="fail_number",
            hover_data=["course_code", "total_students", "fail_rate"],
            color_continuous_scale="Reds",
            labels={
                "course_name": "Course Name",
                "fail_number": "Fail Number",
                "fail_rate": "Fail Rate (%)",
            },
        )
        fig.update_layout(
            xaxis_title="Course Name",
            yaxis_title="Fail Number",
            xaxis_tickangle=-25,
        )
        aura_plotly_chart(fig, use_container_width=True)

    st.markdown("### Course Failure Table")
    table_df = summary_df[
        ["course_code", "course_name", "total_students", "fail_number", "fail_rate"]
    ].rename(
        columns={
            "course_code": "Course Code",
            "course_name": "Course Name",
            "total_students": "Total Students",
            "fail_number": "Fail Number",
            "fail_rate": "Fail Rate",
        }
    )
    st.dataframe(
        table_df.style.format({"Fail Rate": "{:.1f}%"}),
        use_container_width=True,
        hide_index=True,
    )


def dashboard_page():
    st.title("Dashboard")
    if is_advisor_user():
        risk_tab, course_tab = st.tabs(["Risk Overview", "Course Failure Analysis"])
        with risk_tab:
            render_risk_dashboard_overview()
        with course_tab:
            render_course_failure_analysis()
    else:
        render_risk_dashboard_overview()


def risk_trend_page():
    st.title("Risk Trend")
    st.write("Monitor how student risk changes over time.")

    df = load_history()
    if df.empty:
        st.info("No prediction records yet.")
        return

    df["prediction_time"] = pd.to_datetime(df["prediction_time"], errors="coerce")
    df = df.dropna(subset=["prediction_time"]).copy()
    if df.empty:
        st.info("No valid prediction dates are available.")
        return

    risk_order = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk"]
    risk_colors = {
        "Critical Risk": "#b91c1c",
        "High Risk": "#ef4444",
        "Medium Risk": "#f59e0b",
        "Low Risk": "#16a34a",
    }
    df["predicted_risk"] = df["predicted_risk"].apply(normalize_risk_filter_value)
    df = df[df["predicted_risk"].isin(risk_order)].copy()
    df["probability_score"] = pd.to_numeric(df["probability_score"], errors="coerce")
    df = df.dropna(subset=["probability_score"])

    period_col, risk_col, search_col = st.columns([1, 1, 2])
    with period_col:
        period_label = st.selectbox(
            "Trend Period",
            ["Daily", "Weekly", "Monthly"],
            key="risk_trend_period",
        )
    with risk_col:
        risk_filter = st.selectbox(
            "Risk Level",
            ["All"] + risk_order,
            key="risk_trend_risk_filter",
        )
    with search_col:
        search = st.text_input("Search student name or ID", key="risk_trend_search")

    if risk_filter != "All":
        df = df[df["predicted_risk"].eq(risk_filter)]
    if search:
        search_text = search.strip()
        mask = (
            df["student_name"].astype(str).str.contains(
                search_text, case=False, na=False, regex=False
            )
            | df["student_id"].astype(str).str.contains(
                search_text, case=False, na=False, regex=False
            )
        )
        df = df[mask]

    if df.empty:
        st.info("No risk trend records match the selected filters.")
        return

    period_map = {"Daily": "D", "Weekly": "W-SUN", "Monthly": "M"}
    period_format = {
        "Daily": "%Y-%m-%d",
        "Weekly": "Week of %Y-%m-%d",
        "Monthly": "%Y-%m",
    }
    df["trend_period_start"] = (
        df["prediction_time"]
        .dt.to_period(period_map[period_label])
        .dt.start_time
    )
    df["trend_period_label"] = df["trend_period_start"].dt.strftime(
        period_format[period_label]
    )
    st.caption(f"Trend grouped by: {period_label}")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Records", len(df))
    metric_cols[1].metric(
        "Average Risk Probability",
        f"{df['probability_score'].mean():.2f}%",
    )
    high_risk_count = int(
        df["predicted_risk"].isin(["Critical Risk", "High Risk"]).sum()
    )
    metric_cols[2].metric("High/Critical Records", high_risk_count)

    probability_trend = (
        df.groupby(["trend_period_start", "trend_period_label"], as_index=False)["probability_score"]
        .mean()
        .sort_values("trend_period_start")
    )
    fig_probability = px.line(
        probability_trend,
        x="trend_period_label",
        y="probability_score",
        markers=True,
        labels={
            "trend_period_label": "Period",
            "probability_score": "Average Risk Probability (%)",
        },
    )
    fig_probability.update_layout(
        xaxis_title=f"{period_label} Period",
        yaxis_title="Average Risk Probability (%)",
    )
    fig_probability.update_xaxes(type="category")
    st.subheader("Average Risk Probability Over Time")
    aura_plotly_chart(fig_probability, use_container_width=True)

    risk_trend = (
        df.groupby(["trend_period_start", "trend_period_label", "predicted_risk"])
        .size()
        .reset_index(name="Count")
        .sort_values("trend_period_start")
    )
    fig_risk = px.bar(
        risk_trend,
        x="trend_period_label",
        y="Count",
        color="predicted_risk",
        category_orders={"predicted_risk": risk_order},
        color_discrete_map=risk_colors,
        labels={"trend_period_label": "Period", "predicted_risk": "Risk Level"},
    )
    fig_risk.update_layout(xaxis_title=f"{period_label} Period")
    fig_risk.update_xaxes(type="category")
    st.subheader("Risk Level Records Over Time")
    aura_plotly_chart(fig_risk, use_container_width=True)


def model_information_page():
    st.title("Model Information")
    st.write("Model transparency summary for advisors and administrators.")

    summary = get_model_information_summary()

    overview_items = [
        ("Model Version", summary["model_version"]),
        ("Model File", summary["model_file"]),
        ("Algorithm", summary["algorithm"]),
        ("Training Dataset", summary["training_dataset"]),
        ("Training Data Size", f"{summary['dataset_rows']} records"),
        ("Features Used", f"{len(MODEL_FEATURE_NAMES)} features"),
    ]
    render_detail_grid(overview_items)

    training_tab, performance_tab = st.tabs(
        ["Last Training Info", "Model Performance Metrics"]
    )

    with training_tab:
        training_items = [
            ("Last Training / Model Update", summary["last_trained"]),
            ("Training Rows", summary["training_rows"]),
            ("Test Rows", summary["test_rows"]),
            ("Train/Test Split", "80% training / 20% testing"),
            ("Random State", "42"),
        ]
        render_detail_grid(training_items)

    with performance_tab:
        performance = summary["performance"]
        if performance:
            metric_cols = st.columns(len(performance))
            for col, (metric, value) in zip(metric_cols, performance.items()):
                col.metric(metric, value)
        else:
            st.info("Performance metrics are not available in the current runtime.")

        if summary["error"]:
            st.warning(f"Metric calculation note: {summary['error']}")


def manual_prediction_page():
    st.title("Manual Student Prediction")

    st.caption(
        "Enter the student's current academic and personal indicators. "
        "All fields are evaluated together when the prediction is submitted."
    )

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        student_name = st.text_input("Student Name")
    with row_right:
        gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, value=3.0)

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        student_id = st.text_input("Student ID")
    with row_right:
        internet = st.selectbox("Internet Access", ["Yes", "No"])

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        age = st.number_input("Age", min_value=15, max_value=40, value=20)
    with row_right:
        part_time_job = st.selectbox("Part-Time Job", ["No", "Yes"])

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        study_hours = st.number_input(
            "Study Hours per Day", min_value=0.0, max_value=12.0, value=4.0
        )
    with row_right:
        family_problems = st.selectbox("Family Problems", ["No", "Yes"])

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        attendance = st.number_input(
            "Attendance Rate (%)", min_value=0.0, max_value=100.0, value=80.0
        )
    with row_right:
        scholarship = st.selectbox("Scholarship", ["No", "Yes"])

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        assignment_delay = st.number_input(
            "Assignment Delay Days", min_value=0.0, max_value=30.0, value=2.0
        )
    with row_right:
        department = st.selectbox(
            "Department",
            ["Business", "CS", "Engineering", "Science"],
        )

    row_left, row_right = st.columns(2, gap="large")
    with row_left:
        stress = st.number_input(
            "Stress Index (1-10)", min_value=0.0, max_value=10.0, value=5.0
        )
    with row_right:
        semester = st.selectbox(
            "Semester",
            SEMESTER_OPTIONS,
        )

    family_reason = ""
    if family_problems == "Yes":
        family_reason = st.text_area(
            "Describe Family Problem",
            placeholder="Example: financial pressure, family illness, lack of family support...",
        )

    submitted = st.button("Analyze Student Risk", use_container_width=True)

    if not submitted:
        return

    if not student_name.strip() or not student_id.strip():
        st.error("Student name and student ID are required.")
        return

    input_data = make_input_dataframe(
        age,
        study_hours,
        attendance,
        assignment_delay,
        stress,
        internet,
        part_time_job,
        scholarship,
        semester,
        department,
    )
    probability, risk_status, color, advice, factors = predict_student(input_data)
    interventions = generate_interventions(
        risk_status,
        attendance,
        stress,
        assignment_delay,
        study_hours,
        gpa,
        internet,
        family_problems,
        part_time_job,
        factors,
    )
    if family_problems == "Yes" and family_reason.strip():
        interventions.append("Family issue note: " + family_reason.strip())
    ai_suggestions = generate_ai_suggestions(
        risk_status, probability, factors, interventions
    )

    save_prediction(
        "Manual Prediction",
        student_name.strip(),
        student_id.strip(),
        age,
        study_hours,
        attendance,
        assignment_delay,
        stress,
        gpa,
        internet,
        part_time_job,
        family_problems,
        family_reason.strip() if family_problems == "Yes" else "",
        scholarship,
        department,
        semester,
        probability,
        risk_status,
        factors,
        interventions,
        ai_suggestions,
    )

    st.success("Prediction saved successfully.")

    result_tab, advisor_tab, interventions_tab, factors_tab = st.tabs(
        [
            "Prediction Result",
            "AI Advisor Note",
            "Interventions",
            "SHAP Factors",
        ]
    )

    with result_tab:
        render_manual_prediction_result(
            student_name.strip(),
            student_id.strip(),
            semester,
            gpa,
            stress,
            probability,
            risk_status,
            advice,
        )

    with advisor_tab:
        st.markdown(ai_suggestions)

    with interventions_tab:
        if interventions:
            for item in interventions:
                st.write(f"- {item}")
        else:
            st.info("No intervention suggestions available.")

    with factors_tab:
        if factors:
            for factor in factors:
                st.write(f"- {factor}")
        else:
            st.info("No SHAP factors available.")


def upload_csv_page():
    st.title("Upload CSV Data")
    required_columns = [
        "student_name",
        "student_id",
        "age",
        "study_hours",
        "attendance",
        "assignment_delay",
        "stress",
        "gpa",
        "internet",
        "part_time_job",
        "family_problems",
        "scholarship",
        "department",
        "semester",
    ]
    st.info("Required columns: " + ", ".join(required_columns))
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is None:
        return

    df = pd.read_csv(uploaded_file)
    file_signature = f"{uploaded_file.name}:{getattr(uploaded_file, 'size', len(df))}"
    if st.session_state.get("csv_upload_signature") != file_signature:
        st.session_state.csv_upload_signature = file_signature
        st.session_state.csv_prediction_results = pd.DataFrame()
        st.session_state.pop("csv_results_selected_prediction", None)

    st.caption(f"{len(df)} students loaded. Click Analyze CSV Data to create results.")

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing}")
        return

    analyze_clicked = st.button("Analyze CSV Data", use_container_width=True)

    if analyze_clicked:
        results = []
        upload_batch_id = notifications.create_upload_batch_id()
        progress = st.progress(0)
        for index, row in df.iterrows():
            try:
                progress.progress((index + 1) / len(df))
                student_name = str(row["student_name"]).strip()
                student_id = str(row["student_id"]).strip()
                age = float(row["age"])
                study_hours = float(row["study_hours"])
                attendance = float(row["attendance"])
                assignment_delay = float(row["assignment_delay"])
                stress = normalize_stress(row["stress"])
                gpa = float(row["gpa"])
                internet = normalize_yes_no(row["internet"])
                part_time_job = normalize_yes_no(row["part_time_job"])
                family_problems = normalize_yes_no(row["family_problems"])
                family_reason = ""
                if "family_reason" in df.columns and not pd.isna(row["family_reason"]):
                    family_reason = str(row["family_reason"]).strip()
                scholarship = normalize_yes_no(row["scholarship"])
                department = normalize_department(row["department"])
                semester = normalize_semester(row["semester"])

                input_data = make_input_dataframe(
                    age,
                    study_hours,
                    attendance,
                    assignment_delay,
                    stress,
                    internet,
                    part_time_job,
                    scholarship,
                    semester,
                    department,
                )
                probability, risk_status, _, _, factors = predict_student(input_data)
                interventions = generate_interventions(
                    risk_status,
                    attendance,
                    stress,
                    assignment_delay,
                    study_hours,
                    gpa,
                    internet,
                    family_problems,
                    part_time_job,
                    factors,
                )
                if family_problems == "Yes" and family_reason:
                    interventions.append("Family issue note: " + family_reason)
                ai_suggestions = generate_ai_suggestions(
                    risk_status,
                    probability,
                    factors,
                    interventions,
                )

                prediction_id = save_prediction(
                    "Upload CSV",
                    student_name,
                    student_id,
                    age,
                    study_hours,
                    attendance,
                    assignment_delay,
                    stress,
                    gpa,
                    internet,
                    part_time_job,
                    family_problems,
                    family_reason if family_problems == "Yes" else "",
                    scholarship,
                    department,
                    semester,
                    probability,
                    risk_status,
                    factors,
                    interventions,
                    ai_suggestions,
                    upload_batch_id=upload_batch_id,
                )

                results.append(
                    {
                        "prediction_id": prediction_id,
                        "upload_batch_id": upload_batch_id,
                        "student_name": student_name,
                        "student_id": student_id,
                        "age": age,
                        "study_hours": study_hours,
                        "attendance": attendance,
                        "assignment_delay": assignment_delay,
                        "predicted_risk": risk_status,
                        "probability_score": round(probability, 2),
                        "semester": semester,
                        "gpa": gpa,
                        "stress_index": stress,
                        "internet": internet,
                        "part_time_job": part_time_job,
                        "family_problems": family_problems,
                        "family_reason": family_reason,
                        "scholarship": scholarship,
                        "department": department,
                        "top_factors": "; ".join(factors),
                        "interventions": " | ".join(interventions),
                        "ai_suggestions": ai_suggestions,
                    }
                )
            except Exception as error:
                results.append(
                    {
                        "student_name": row.get("student_name", "Unknown"),
                        "student_id": row.get("student_id", "Unknown"),
                        "upload_batch_id": upload_batch_id,
                        "predicted_risk": "Failed",
                        "probability_score": 0,
                        "error": str(error),
                    }
                )

        progress.empty()
        results_df = pd.DataFrame(results)
        st.session_state.csv_prediction_results = results_df
        st.session_state.pop("csv_results_selected_prediction", None)
        st.success(f"{len(results_df)} rows processed. Upload batch: {upload_batch_id}")
    else:
        results_df = st.session_state.get("csv_prediction_results", pd.DataFrame())
        if results_df.empty:
            return

    render_student_prediction_list(results_df, "csv_results", page_size=15)
    st.download_button(
        "Download Results CSV",
        results_df.to_csv(index=False).encode("utf-8"),
        "aura_prediction_results.csv",
        "text/csv",
        use_container_width=True,
    )


def course_results_upload_page():
    st.title("Course Results Upload")
    st.caption(
        "Upload course result datasets for Course Failure Analysis. Final and Resit uploads use the same semester; Resit updates the latest current result."
    )

    required_columns = [
        "student_id",
        "student_name",
        "course_code",
        "course_name",
        "academic_year",
        "semester_term",
        "upload_type",
        "course_mark",
    ]
    st.info("Required columns: " + ", ".join(required_columns))

    sample_file = PROJECT_DIR / "course_results_2026_february_sample.csv"
    if sample_file.exists():
        st.download_button(
            "Download Sample 2026 February Course Results CSV",
            sample_file.read_bytes(),
            "course_results_2026_february_sample.csv",
            "text/csv",
            use_container_width=True,
        )

    with get_conn() as conn:
        catalog_count = conn.execute("SELECT COUNT(*) FROM course_catalog").fetchone()[0]
    st.caption(f"Course catalog loaded: {catalog_count} courses.")

    filter_cols = st.columns(3)
    with filter_cols[0]:
        academic_year = st.number_input(
            "Academic Year",
            min_value=2020,
            max_value=2035,
            value=2026,
            step=1,
            key="course_upload_year",
        )
    with filter_cols[1]:
        semester_term = st.selectbox(
            "Semester Term",
            ["February", "April", "September"],
            key="course_upload_semester",
        )
    with filter_cols[2]:
        upload_type = st.selectbox(
            "Upload Type",
            course_failures.VALID_UPLOAD_TYPES,
            key="course_upload_type",
        )

    uploaded_file = st.file_uploader(
        "Upload course_results.csv",
        type=["csv"],
        key="course_results_uploader",
    )
    if uploaded_file is None:
        return

    try:
        df = pd.read_csv(uploaded_file, dtype=str, keep_default_na=False).fillna("")
    except Exception as error:
        st.error(f"Could not read CSV: {error}")
        return

    st.caption(f"{len(df)} course result rows loaded.")
    st.dataframe(df.head(20), use_container_width=True)

    if st.button("Save Course Results", type="primary", use_container_width=True):
        try:
            with get_conn() as conn:
                raw_count, current_count = course_failures.save_course_results(
                    conn,
                    df,
                    st.session_state.user_email,
                    int(academic_year),
                    semester_term,
                    upload_type,
                )
            st.success(
                f"Saved {raw_count} raw result row(s). Updated {current_count} current result row(s)."
            )
        except Exception as error:
            st.error(f"Course result upload failed: {error}")


def prediction_history_page(
    page_title="Student List",
    key_prefix="history_results",
    export_label="Export Student List CSV",
    export_filename="aura_student_list.csv",
):
    st.title(page_title)
    df = load_history()
    if df.empty:
        st.info("No students in the list yet.")
        return

    total_records = len(df)
    df["predicted_risk"] = df["predicted_risk"].apply(normalize_risk_filter_value)

    upload_batches = load_upload_batches()
    selected_batch = "All"
    if upload_batches:
        selected_batch = st.selectbox(
            "Upload Batch",
            ["All"] + upload_batches,
            key=f"{key_prefix}_upload_batch_filter",
        )
        if selected_batch != "All":
            df = df[df["upload_batch_id"].astype(str).eq(selected_batch)]

        ensure_pending_notifications_for_df(df)

        send_col, hint_col = st.columns([1.4, 3])
        with send_col:
            send_disabled = selected_batch == "All" or not can_send_notifications()
            if st.button(
                "Send All Pending Alerts",
                key=f"{key_prefix}_send_pending_alerts",
                use_container_width=True,
                disabled=send_disabled,
            ):
                sent_count, failed_count, remaining_count, messages = send_pending_notifications_for_batch(selected_batch)
                if sent_count:
                    st.success(f"{sent_count} pending alert(s) sent.")
                if failed_count:
                    st.error(f"{failed_count} alert(s) failed.")
                    if messages:
                        st.caption(messages[0])
                if remaining_count:
                    st.info(
                        f"{remaining_count} pending alert(s) were not sent in this click. "
                        f"To avoid email rate limits, AURA sends up to {BULK_NOTIFICATION_SEND_LIMIT} alerts at a time."
                    )
                st.rerun()
        with hint_col:
            if not can_send_notifications():
                st.caption("Only Academic Advisors can send student notifications.")
            elif selected_batch == "All":
                st.caption("Select one upload batch before sending pending alerts.")
            else:
                st.caption(
                    "Bulk sending only applies to pending High Risk and Critical Risk students in this batch. "
                    f"For email safety, AURA sends up to {BULK_NOTIFICATION_SEND_LIMIT} alerts per click."
                )

    ensure_pending_notifications_for_df(df)

    risk_filter = st.selectbox(
        "Filter by Risk Level",
        ["All", "Critical Risk", "High Risk", "Medium Risk", "Low Risk"],
        key=f"{key_prefix}_risk_filter",
    )
    search = st.text_input("Search by student name or ID", key=f"{key_prefix}_search")

    filter_signature = f"{selected_batch}|{risk_filter}|{search.strip().lower()}"
    signature_key = f"{key_prefix}_filter_signature"
    if st.session_state.get(signature_key) != filter_signature:
        st.session_state[signature_key] = filter_signature
        st.session_state[f"{key_prefix}_page"] = 1

    if risk_filter != "All":
        df = df[df["predicted_risk"].eq(risk_filter)]

    if search:
        mask = (
            df["student_name"].astype(str).str.contains(
                search, case=False, na=False, regex=False
            )
            | df["student_id"].astype(str).str.contains(
                search, case=False, na=False, regex=False
            )
        )
        df = df[mask]

    if df.empty:
        st.info("No matching students found.")
        return

    st.caption(f"Showing {len(df)} of {total_records} student records.")
    render_student_prediction_list(df, key_prefix, page_size=15)
    st.download_button(
        export_label,
        df.to_csv(index=False).encode("utf-8"),
        export_filename,
        "text/csv",
        use_container_width=True,
    )


def student_timeline_page():
    st.title("Student Timeline")
    student_id = st.text_input("Enter Student ID", key="timeline_student_id")
    if st.button("Search", use_container_width=True):
        st.session_state.timeline_search_clicked = True

    if not st.session_state.get("timeline_search_clicked"):
        return

    if not student_id.strip():
        st.error("Please enter a student ID.")
        return

    with get_conn() as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM prediction_history
            WHERE student_id = ?
            ORDER BY prediction_time ASC
            """,
            conn,
            params=(student_id.strip(),),
        )

    if df.empty:
        st.warning("No history found for this student.")
        return

    df["predicted_risk"] = df["predicted_risk"].apply(normalize_risk_filter_value)
    st.caption(f"{len(df)} timeline records found for Student ID {student_id.strip()}.")
    render_student_prediction_list(
        df.sort_values("prediction_time", ascending=False),
        "timeline_results",
        page_size=10,
    )

    chart_df = df[["prediction_time", "probability_score"]].set_index(
        "prediction_time"
    )
    st.subheader("Risk Probability Trend")
    st.line_chart(chart_df)


def render_delete_account_confirmation():
    candidate = st.session_state.get("delete_account_candidate")
    if not candidate:
        return

    def confirmation_content():
        name = clean_display_value(candidate.get("name"), "this user")
        email = clean_display_value(candidate.get("email"), "")
        st.warning(f"Remove account for {name}?")
        st.write(f"Email: {email}")
        st.caption(
            "This removes only the registered login account. "
            "Approved access and student prediction records will not be deleted."
        )
        cancel_col, remove_col = st.columns(2)
        with cancel_col:
            if st.button("Cancel", use_container_width=True, key="cancel_delete_account"):
                st.session_state.delete_account_candidate = None
                st.rerun()
        with remove_col:
            if st.button(
                "Remove Account",
                use_container_width=True,
                key="confirm_delete_account",
                type="primary",
            ):
                ok, message = delete_registered_account(email)
                st.session_state.delete_account_candidate = None
                st.session_state.manage_users_message = message
                if not ok:
                    st.session_state.manage_users_message = f"Could not remove account: {message}"
                st.rerun()

    if hasattr(st, "dialog"):
        @st.dialog("Confirm Account Removal")
        def delete_account_dialog():
            confirmation_content()

        delete_account_dialog()
    else:
        st.error("Confirm Account Removal")
        confirmation_content()


def manage_users_page():
    st.title("Manage Users")
    if not is_administrator_user():
        st.error("Access denied.")
        return

    with get_conn() as conn:
        users_df = pd.read_sql_query(
            """
            SELECT id, name, email, staff_id, role, first_login, created_at
            FROM users
            ORDER BY created_at DESC
            """,
            conn,
        )
    if st.session_state.manage_users_message:
        message = st.session_state.manage_users_message
        if message.startswith("Could not"):
            st.error(message)
        else:
            st.success(message)
        st.session_state.manage_users_message = ""

    st.subheader("Registered Accounts")
    if users_df.empty:
        st.info("No registered accounts found.")
    else:
        header_cols = st.columns([2, 3, 1.4, 1.4, 1.2])
        header_cols[0].markdown("**Name**")
        header_cols[1].markdown("**Email**")
        header_cols[2].markdown("**ID Number**")
        header_cols[3].markdown("**Role**")
        header_cols[4].markdown("**Remove**")

        for _, row in users_df.iterrows():
            row_cols = st.columns([2, 3, 1.4, 1.4, 1.2])
            row_cols[0].write(clean_display_value(row.get("name"), "Unknown"))
            row_cols[1].write(clean_display_value(row.get("email"), ""))
            row_cols[2].write(clean_display_value(row.get("staff_id"), "N/A"))
            row_cols[3].write(clean_display_value(row.get("role"), ""))
            with row_cols[4]:
                email = clean_display_value(row.get("email"), "")
                disabled = email.lower() == st.session_state.user_email.lower()
                if st.button(
                    "Remove",
                    key=f"remove_account_{row.get('id')}",
                    use_container_width=True,
                    disabled=disabled,
                ):
                    st.session_state.delete_account_candidate = {
                        "name": clean_display_value(row.get("name"), "Unknown"),
                        "email": email,
                    }
                    st.rerun()

    st.subheader("Account Controls")
    with st.expander("Add User", expanded=False):
        add_col1, add_col2 = st.columns(2)
        with add_col1:
            new_email = st.text_input("Email", key="approved_new_email")
        with add_col2:
            new_staff_id = st.text_input("ID Number", key="approved_new_staff_id")
            st.caption("The user will choose Administrator or Advisor during registration.")
        if st.button("Add User", use_container_width=True):
            inferred_role = infer_staff_role_from_id(new_staff_id)
            account_ok, account_message = (
                validate_institutional_account(new_email, new_staff_id, inferred_role)
                if inferred_role
                else (False, "ID Number must use ADM001 or ADV001 format.")
            )
            if not new_email or not new_staff_id:
                st.error("Please fill all fields.")
            elif not account_ok:
                st.error(account_message)
            elif add_authorized_user(new_email, new_staff_id):
                st.session_state.manage_users_message = (
                    f"User added as {role_display_name(inferred_role)}. "
                    "They can now register an account."
                )
                st.rerun()
            else:
                st.error("This ID number is already used by another user.")

    with st.expander("Change Role", expanded=False):
        role_email = st.text_input("User Email", key="approved_role_email")
        role_value = st.selectbox(
            "New Role",
            ["administrator", "advisor"],
            key="approved_role_value",
        )
        if st.button("Update User Role", use_container_width=True):
            ok, message = update_authorized_user_role(role_email, role_value)
            if ok:
                st.session_state.manage_users_message = (
                    f"{message} The user should log in with the new institutional email next time."
                )
                st.rerun()
            else:
                st.session_state.manage_users_message = f"Could not update role: {message}"
                st.rerun()

    render_delete_account_confirmation()


def audit_logs_page():
    st.title("Audit Logs")
    if not is_administrator_user():
        st.error("Access denied. Administrator users only.")
        return

    logs_df = load_audit_logs()
    if logs_df.empty:
        st.info("No audit logs recorded yet.")
        return

    total_logs = len(logs_df)
    logs_df["_malaysia_time_sort"] = logs_df["created_at"].apply(
        audit_timestamp_to_malaysia
    )
    logs_df["Malaysia Time (MYT)"] = logs_df["created_at"].apply(
        format_audit_malaysia_time
    )
    logs_df["action_type"] = logs_df["action_type"].apply(normalize_audit_action)
    logs_df["action_status"] = logs_df["action_status"].apply(normalize_audit_status)
    action_options = ["All"] + sorted(
        logs_df["action_type"].dropna().astype(str).unique().tolist()
    )
    status_options = ["All", "Success", "Fail"]

    filter_col, status_col, search_col = st.columns([1.2, 1.2, 2])
    with filter_col:
        action_filter = st.selectbox("Action Type", action_options)
    with status_col:
        status_filter = st.selectbox("Status", status_options)
    with search_col:
        search = st.text_input("Search email or details")

    if action_filter != "All":
        logs_df = logs_df[logs_df["action_type"].astype(str).eq(action_filter)]
    if status_filter != "All":
        logs_df = logs_df[logs_df["action_status"].astype(str).eq(status_filter)]
    if search:
        search_text = search.strip()
        mask = (
            logs_df["user_email"].astype(str).str.contains(
                search_text, case=False, na=False, regex=False
            )
            | logs_df["action_details"].astype(str).str.contains(
                search_text, case=False, na=False, regex=False
            )
        )
        logs_df = logs_df[mask]

    if logs_df.empty:
        st.info("No audit logs match the selected filters.")
        return

    logs_df = logs_df.sort_values(
        by=["_malaysia_time_sort", "log_id"],
        ascending=[False, False],
        na_position="last",
    )
    audit_display_columns = [
        "Malaysia Time (MYT)",
        "user_email",
        "user_role",
        "action_type",
        "action_status",
        "action_details",
    ]
    audit_display_df = logs_df[audit_display_columns].rename(
        columns={
            "user_email": "User Email",
            "user_role": "Role",
            "action_type": "Action",
            "action_status": "Status",
            "action_details": "Details",
        }
    )

    st.caption(f"Showing {len(logs_df)} of {total_logs} audit log records.")
    st.dataframe(
        audit_display_df,
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Export Audit Logs CSV",
        audit_display_df.to_csv(index=False).encode("utf-8"),
        "aura_audit_logs.csv",
        "text/csv",
        use_container_width=True,
    )


def route_app():
    init_session()

    if st.session_state.page == "home":
        render_page_tools("page:home", show_back_to_top=False)
        home_page()
        return
    if st.session_state.page == "login":
        render_page_tools("page:login", show_back_to_top=False)
        login_page()
        return
    if st.session_state.page == "forgot_password":
        render_page_tools("page:forgot_password", show_back_to_top=False)
        forgot_password_page()
        return
    if st.session_state.page == "reset_password":
        render_page_tools("page:reset_password", show_back_to_top=False)
        reset_password_page()
        return
    if st.session_state.page == "register":
        render_page_tools("page:register", show_back_to_top=False)
        register_page()
        return
    if st.session_state.page == "otp":
        render_page_tools("page:otp", show_back_to_top=False)
        otp_page()
        return

    if not st.session_state.logged_in:
        st.session_state.page = "home"
        render_page_tools("page:home", show_back_to_top=False)
        home_page()
        return

    if st.session_state.page == "chatbot":
        render_page_tools("page:chatbot")
        chatbot_page()
        return
    if st.session_state.page == "prediction_detail":
        detail_record = st.session_state.get("prediction_detail_record") or {}
        detail_key = (
            clean_display_value(detail_record.get("prediction_id"))
            or clean_display_value(detail_record.get("student_id"))
            or "selected"
        )
        render_page_tools(f"page:prediction_detail:{detail_key}")
        prediction_detail_page()
        return

    selected = sidebar_menu()
    render_page_tools(f"nav:{selected}")
    if selected == "Support Questionnaire":
        student_questionnaire_page()
    elif selected == "Dashboard":
        dashboard_page()
    elif selected == "Model Information":
        model_information_page()
    elif selected == "Manual Prediction":
        manual_prediction_page()
    elif selected == "Upload CSV":
        upload_csv_page()
    elif selected == "Course Results Upload":
        course_results_upload_page()
    elif selected == "Student List":
        prediction_history_page()
    elif selected == "Student Timeline":
        student_timeline_page()
    elif selected == "Risk Trend":
        risk_trend_page()
    elif selected == "Manage Users":
        manage_users_page()
    elif selected == "Audit Logs":
        audit_logs_page()

    show_chat_fab()


route_app()

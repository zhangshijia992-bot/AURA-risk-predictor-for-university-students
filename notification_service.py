import smtplib
import urllib.parse
import re
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage


NOTIFIABLE_RISKS = {"High Risk", "Critical Risk"}
APP_TIMEZONE = timezone(timedelta(hours=8))


def app_now():
    return datetime.now(APP_TIMEZONE).replace(tzinfo=None)


def app_now_text():
    return app_now().strftime("%Y-%m-%d %H:%M:%S")


def create_upload_batch_id(now=None):
    timestamp = (now or app_now()).strftime("%Y%m%d_%H%M%S")
    return f"BATCH_{timestamp}"


def is_notifiable_risk(risk_level):
    return str(risk_level or "").strip() in NOTIFIABLE_RISKS


def normalize_status(status):
    text = str(status or "").strip()
    return text if text else "Pending"


def email_settings_from_secrets(secrets):
    email_section = secrets.get("email", {})
    return {
        "smtp_host": str(email_section.get("smtp_host", "")).strip(),
        "smtp_port": int(str(email_section.get("smtp_port", "587")).strip() or 587),
        "smtp_user": str(email_section.get("smtp_user", "")).strip(),
        "smtp_password": str(email_section.get("smtp_password", "")).strip(),
        "from_email": str(email_section.get("from_email", "")).strip(),
    }


def find_student_email(conn, student_id):
    student_id = str(student_id or "").strip()
    if not student_id:
        return ""

    row = conn.execute(
        """
        SELECT student_email, preferred_contact_details, preferred_contact_method
        FROM student_questionnaire
        WHERE upper(student_id) = upper(?)
        ORDER BY submitted_at DESC, questionnaire_id DESC
        LIMIT 1
        """,
        (student_id,),
    ).fetchone()
    if row:
        preferred_details = str(row[1] or "").strip()
        email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", preferred_details)
        if email_match:
            return email_match.group(0).lower()
        if row[0]:
            return str(row[0]).strip().lower()

    row = conn.execute(
        """
        SELECT student_email
        FROM student_contacts
        WHERE upper(student_id) = upper(?)
          AND lower(COALESCE(status, 'active')) = 'active'
        LIMIT 1
        """,
        (student_id,),
    ).fetchone()
    if row and row[0]:
        return str(row[0]).strip().lower()

    row = conn.execute(
        """
        SELECT email
        FROM users
        WHERE role = 'student' AND upper(staff_id) = upper(?)
        ORDER BY created_at DESC
        LIMIT 1
        """,
        (student_id,),
    ).fetchone()
    if row and row[0]:
        return str(row[0]).strip().lower()

    return ""


def latest_notification_for_prediction(conn, prediction_id):
    return conn.execute(
        """
        SELECT notification_id, notification_type, send_status, sent_at, sent_by,
               message_subject, message_body, student_email
        FROM notification_logs
        WHERE prediction_id = ?
        ORDER BY
            CASE send_status
                WHEN 'Pending' THEN 0
                WHEN 'Failed' THEN 1
                WHEN 'Sent' THEN 2
                WHEN 'Resent' THEN 3
                ELSE 4
            END DESC,
            COALESCE(sent_at, created_at) DESC,
            notification_id DESC
        LIMIT 1
        """,
        (prediction_id,),
    ).fetchone()


def latest_notification_status(conn, prediction_id, risk_level):
    if not is_notifiable_risk(risk_level):
        return "Not Required"

    row = latest_notification_for_prediction(conn, prediction_id)
    if row is None:
        return "Pending"
    return normalize_status(row[2])


def ensure_pending_initial_notification(conn, prediction_record):
    risk_level = str(prediction_record.get("predicted_risk", "")).strip()
    if not is_notifiable_risk(risk_level):
        return None

    prediction_id = prediction_record.get("prediction_id")
    existing = conn.execute(
        """
        SELECT notification_id
        FROM notification_logs
        WHERE prediction_id = ?
          AND notification_type = 'Initial Alert'
          AND send_status IN ('Pending', 'Sent', 'Resent')
        LIMIT 1
        """,
        (prediction_id,),
    ).fetchone()
    if existing:
        return existing[0]

    student_email = find_student_email(conn, prediction_record.get("student_id"))
    subject = prediction_record.get("message_subject")
    body = prediction_record.get("message_body")
    if not subject or not body:
        subject, body = build_system_message(prediction_record)
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
            prediction_record.get("upload_batch_id", ""),
            prediction_id,
            prediction_record.get("student_id", ""),
            student_email,
            risk_level,
            "Initial Alert",
            subject,
            body,
            app_now_text(),
        ),
    )
    return cursor.lastrowid


def _parse_json_list(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    try:
        import json

        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
    except Exception:
        pass
    return [text]


def _clean_factor_text(value):
    text = str(value or "")
    for phrase in ["increased risk", "reduced risk", ":", ";"]:
        text = text.replace(phrase, " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _student_friendly_factor_summary(record):
    raw_factors = str(record.get("top_factors", "") or "")
    normalized = raw_factors.lower()
    summaries = []
    if "attendance" in normalized:
        summaries.append("attendance pattern")
    if "assignment" in normalized or "delay" in normalized:
        summaries.append("assignment submission timing")
    if "study" in normalized:
        summaries.append("weekly study routine")
    if "stress" in normalized:
        summaries.append("reported stress level")
    if "gpa" in normalized:
        summaries.append("recent academic performance")
    if not summaries:
        cleaned = [_clean_factor_text(item) for item in raw_factors.split(";")]
        summaries = [item for item in cleaned if item][:3]
    if not summaries:
        return ""
    return (
        "AURA noticed that the following areas may need attention this semester: "
        + ", ".join(summaries[:4])
        + "."
    )


def _build_practical_steps(record, questionnaire=None, course_recommendations=None):
    raw_factors = str(record.get("top_factors", "") or "").lower()
    steps = []

    if "attendance" in raw_factors:
        steps.append(
            "Check your attendance for each module this week and attend the next class or tutorial if you have recently missed sessions."
        )
    if "assignment" in raw_factors or "delay" in raw_factors:
        steps.append(
            "List all upcoming assignments with their deadlines and start with the closest task first, even if it is only a small draft."
        )
    if "study" in raw_factors:
        steps.append(
            "Set two or three fixed study blocks this week for the modules that need more time."
        )
    if "stress" in raw_factors:
        steps.append(
            "If stress is affecting your study routine, reduce the plan into smaller daily tasks and consider asking for academic or wellbeing support."
        )
    if "gpa" in raw_factors:
        steps.append(
            "Review the topics or assessments with the lowest marks and prepare one question to ask during consultation or class."
        )

    for recommendation in course_recommendations or []:
        if len(steps) >= 5:
            break
        steps.append(str(recommendation).strip())

    if questionnaire:
        preferred_support = _parse_json_list(questionnaire.get("preferred_support"))
        barriers = _parse_json_list(questionnaire.get("main_barrier"))
        contact_methods = _parse_json_list(questionnaire.get("preferred_contact_method"))
        contact_details = str(questionnaire.get("preferred_contact_details") or "").strip()

        if preferred_support:
            steps.append(
                "Use the support option you selected in AURA: "
                + ", ".join(preferred_support[:2])
                + "."
            )
        if barriers:
            steps.append(
                "If your main difficulty is "
                + ", ".join(barriers[:1]).lower()
                + ", prepare a short message explaining the issue before contacting the school support channel."
            )
        if contact_methods:
            contact_line = "Your preferred contact method in AURA is " + ", ".join(contact_methods[:2])
            if contact_details:
                contact_line += f" ({contact_details})"
            contact_line += ". Please use this method if you need to ask for help or clarify the next step."
            steps.append(contact_line)

    if not steps:
        steps = [
            "Review your latest attendance, assignment deadlines, and study schedule for this week.",
            "Choose one module that feels most difficult and spend one focused study session on it first.",
            "If you are unsure what to do next, prepare a short message for your lecturer, programme office, or student support channel.",
        ]

    unique_steps = []
    seen = set()
    for step in steps:
        step = re.sub(r"\s+", " ", str(step)).strip()
        if step and step.lower() not in seen:
            seen.add(step.lower())
            unique_steps.append(step)
    return unique_steps[:6]


def build_system_message(record, questionnaire=None, course_recommendations=None, notification_type="Initial Alert"):
    risk_level = str(record.get("predicted_risk", "")).strip()
    student_name = str(record.get("student_name", "Student") or "Student").strip()
    subject = "AURA Academic Support Reminder"

    opening = (
        "AURA has noticed that you may benefit from some additional academic support this semester."
        if risk_level == "Critical Risk"
        else "AURA has noticed a few areas where early support may help your academic progress."
    )

    factor_summary = _student_friendly_factor_summary(record)
    practical_steps = _build_practical_steps(record, questionnaire, course_recommendations)
    steps_text = "\n".join(f"{index}. {step}" for index, step in enumerate(practical_steps, start=1))

    body = f"""Dear {student_name},

This is a supportive academic reminder from the AURA system.

{opening}

This message is not a punishment or final judgement. It is only intended to help you notice possible study issues earlier and take small, manageable actions.

{factor_summary}

Suggested next steps:
{steps_text}

If you are not sure who to contact, you may start with your module lecturer, programme office, student support centre, or the official communication channel normally used by your school.

Please take this as an early support reminder rather than a negative label. A small action this week can still make a meaningful difference.

Regards,
AURA Student Support System
"""
    if notification_type == "Follow-up Reminder":
        subject = "AURA Academic Support Follow-up Reminder"
        body = body.replace(
            "This is a supportive academic reminder from the AURA system.",
            "This is a follow-up academic support reminder from the AURA system.",
        )
    return subject, body


def send_email(settings, receiver_email, subject, body):
    missing = [
        key
        for key in ["smtp_host", "smtp_user", "smtp_password", "from_email"]
        if not settings.get(key)
    ]
    if missing:
        return False, "Missing email setting(s): " + ", ".join(missing)
    if not receiver_email:
        return False, "Student email was not found."

    message = EmailMessage()
    message["From"] = settings["from_email"]
    message["To"] = receiver_email
    message["Subject"] = subject
    message.set_content(body)

    try:
        smtp_host = settings["smtp_host"]
        smtp_port = int(settings["smtp_port"])
        timeout_seconds = 60

        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=timeout_seconds) as server:
                server.login(settings["smtp_user"], settings["smtp_password"])
                server.send_message(message)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=timeout_seconds) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(settings["smtp_user"], settings["smtp_password"])
                server.send_message(message)
        return True, "Email sent."
    except Exception as error:
        return False, f"{type(error).__name__}: {error}"


def whatsapp_link(phone_or_handle, message_body):
    cleaned = "".join(ch for ch in str(phone_or_handle or "") if ch.isdigit())
    encoded = urllib.parse.quote(message_body)
    if cleaned:
        return f"https://wa.me/{cleaned}?text={encoded}"
    return f"https://wa.me/?text={encoded}"

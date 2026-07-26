import streamlit as st
import pandas as pd
import sqlite3
#academic advisor pages
# =====================================================
# ADVISOR SIDEBAR
# =====================================================

def advisor_sidebar():

    with st.sidebar:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135755.png",
            width=80
        )

        st.markdown("### Academic Advisor")

        st.write(
            st.session_state.get(
                "user_email",
                "advisor@aura.com"
            )
        )

        st.markdown("---")

        selected = st.radio(

            "Navigation",

            [

                "Advisor Dashboard",

                "Student List",

                "Prediction History",

                "Risk Trend"
            ]
        )

        st.markdown("---")

        if st.button("Logout"):

            st.session_state.clear()

            st.rerun()

    return selected

# =====================================================
# ADVISOR MAIN SYSTEM
# =====================================================

def advisor_system():

    role = str(
        st.session_state.get(
            "user_role",
            ""
        )
    ).strip().lower()

    # =========================================
    # ADVISOR ONLY
    # =========================================

    if role != "advisor":

        st.error(
            "Access Denied - Advisor Only"
        )

        return

    # =========================================
    # STUDENT DETAILS PAGE
    # =========================================

    if st.session_state.get("page") == "student_details":

        student_details_page()

        return

    # =========================================
    # NORMAL SIDEBAR PAGES
    # =========================================

    selected = advisor_sidebar()

    if selected == "Advisor Dashboard":

        advisor_dashboard_page()

    elif selected == "Student List":

        advisor_student_list_page()

    elif selected == "Prediction History":

        advisor_prediction_history_page()

    elif selected == "Risk Trend":

        advisor_risk_trend_page()

# =====================================================
# ADVISOR DASHBOARD
# =====================================================

def advisor_dashboard_page():

    st.title(
        "Advisor Dashboard"
    )

    st.write(
        "Monitor student dropout risks and intervention status."
    )

    conn = sqlite3.connect(
        "users.db"
    )

    try:

        df = pd.read_sql_query(

            """

            SELECT *

            FROM prediction_history

            """,

            conn
        )

    except:

        st.warning(
            "No prediction history found."
        )

        return

    conn.close()

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    high_count = len(

        df[
            df["predicted_risk"] == "High Risk"
        ]
    )

    medium_count = len(

        df[
            df["predicted_risk"] == "Medium Risk"
        ]
    )

    safe_count = len(

        df[
            df["predicted_risk"] == "Safe"
        ]
    )

    col1.metric(
        "High Risk",
        high_count
    )

    col2.metric(
        "Medium Risk",
        medium_count
    )

    col3.metric(
        "Safe",
        safe_count
    )

    st.markdown("---")

    risk_counts = pd.DataFrame({

        "Risk": [

            "High Risk",
            "Medium Risk",
            "Safe"
        ],

        "Count": [

            high_count,
            medium_count,
            safe_count
        ]
    })

    st.bar_chart(

        risk_counts.set_index(
            "Risk"
        )
    )


def advisor_student_list_page():

    st.title("Student List")

    st.write(
        "View all predicted student risk records."
    )

    conn = sqlite3.connect("users.db")

    advisor_df = pd.read_sql_query("""

        SELECT

            student_name,
            student_id,
            semester,
            gpa,
            stress_index,
            predicted_risk,
            probability_score,
            prediction_time

        FROM prediction_history

        ORDER BY prediction_time DESC

    """, conn)

    conn.close()

    # =================================================
    # EMPTY TABLE
    # =================================================

    if advisor_df.empty:

        st.info(
            "No prediction records found."
        )

        return

    # =================================================
    # FILTER
    # =================================================

    risk_filter = st.selectbox(

    "Filter Risk Level",

    [

        "All",
        "High Risk",
        "Medium Risk",
        "Low Risk"

    ]

)

    # =========================================
    # FILTER
    # =========================================

    if risk_filter != "All":

     advisor_df = advisor_df[

        advisor_df["predicted_risk"]

        .astype(str)

        .str.strip()

        .str.lower()

        == risk_filter.lower()

    ]

    # =========================================
    # SEARCH
    # =========================================

    search = st.text_input(

        "Search Student"

    )

    if search:

        search = str(search).strip()

        advisor_df = advisor_df[

            advisor_df["student_name"]

            .astype(str)

            .str.contains(

                search,

                case=False,

                na=False

            )

            |

            advisor_df["student_id"]

            .astype(str)

            .str.contains(

                search,

                case=False,

                na=False

            )

        ]
# =================================================
# TABLE
# =================================================

    st.markdown("---")

    header1, header2, header3, header4, header5, header6, header7 = st.columns(
        [2,2,2,2,2,2,1.5]
    )

    with header1:
        st.write("Student Name")

    with header2:
        st.write("Student ID")

    with header3:
        st.write("Semester")

    with header4:
        st.write("GPA")

    with header5:
        st.write("Stress")

    with header6:
        st.write("Risk")

    with header7:
        st.write("")

    st.divider()

    # =========================================
    # REMOVE DUPLICATE STUDENTS
    # =========================================

    advisor_df = advisor_df.drop_duplicates(
        subset=["student_id"],
        keep="first"
    )

    # =========================================
    # STUDENT ROWS
    # =========================================

    for index, row in advisor_df.iterrows():

        col1, col2, col3, col4, col5, col6, col7 = st.columns(
            [2,2,2,2,2,2,1.5]
        )

        # -------------------------------------
        # NAME
        # -------------------------------------

        with col1:
            st.write(
                str(row["student_name"])
            )

        # -------------------------------------
        # ID
        # -------------------------------------

        with col2:
            st.write(
                str(row["student_id"])
            )

        # -------------------------------------
        # SEMESTER
        # -------------------------------------

        with col3:
            st.write(
                str(row["semester"])
            )

        # -------------------------------------
        # GPA
        # -------------------------------------

        with col4:
            st.write(
                round(float(row["gpa"]), 2)
            )

        # -------------------------------------
        # STRESS
        # -------------------------------------

        with col5:
            st.write(
                round(float(row["stress_index"]), 2)
            )

        # -------------------------------------
        # RISK
        # -------------------------------------

        with col6:
            risk = str(
                row["predicted_risk"]
            )

            if risk == "High Risk":
                st.error(risk)
            elif risk == "Medium Risk":
                st.warning(risk)
            else:
                st.success(risk)

        # -------------------------------------
        # VIEW BUTTON
        # -------------------------------------

        with col7:

               view_clicked = st.button(
                     "View",
                     key=f"view_{index}"
    )

        if view_clicked:

                st.session_state.selected_student_id = row["student_id"]

                st.session_state.page = "student_details"

                st.rerun()
        st.divider()

    # =================================================
    # QUICK STATS
    # =================================================

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "High Risk",
            len(
                advisor_df[
                    advisor_df["predicted_risk"]
                    == "High Risk"
                ]
            )
        )

    with col2:

        st.metric(

            "Medium Risk",

            len(

                advisor_df[

                    advisor_df["predicted_risk"]

                    == "Medium Risk"

                ]

            )

        )

    with col3:

        st.metric(

            "Low Risk",

            len(

                advisor_df[

                    advisor_df["predicted_risk"]

                    == "Low Risk"

                ]

            )

        )
def advisor_student_details_page():

    st.markdown("---")

    st.title(
        "Student Details"
    )

    student_id = st.session_state.get(
        "selected_student"
    )

    conn = sqlite3.connect(
        "users.db"
    )

    df = pd.read_sql_query(

        f"""

        SELECT *

        FROM prediction_history

        WHERE student_id = '{student_id}'

        ORDER BY id DESC

        LIMIT 1

        """,

        conn
    )

    conn.close()

    if len(df) == 0:

        st.error(
            "Student not found."
        )

        return

    student = df.iloc[0]

    st.subheader(
        student["student_name"]
    )

    st.write(
        f"Student ID: {student['student_id']}"
    )

    st.write(
        f"Department: {student['department']}"
    )

    st.write(
        f"Semester: {student['semester']}"
    )

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Risk Probability",

        round(
            student["probability"],
            2
        )
    )

    col2.metric(

        "Risk Status",

        student["risk_status"]
    )

    col3.metric(

        "Attendance",

        student["attendance"]
    )

    st.markdown("---")

    st.subheader(
        "Explainable AI Analysis"
    )

    drivers = pd.DataFrame({

        "Factor": [

            "Attendance",
            "Stress",
            "Assignment Delay",
            "Study Hours"
        ],

        "Impact": [

            0.82,
            0.63,
            0.58,
            -0.32
        ]
    })

    st.bar_chart(

        drivers.set_index(
            "Factor"
        )
    )

    st.info(
        "Low attendance rate contributes significantly to dropout risk."
    )

    st.warning(
        "Recommend weekly advisor monitoring."
    )

    st.warning(
        "Recommend improving attendance consistency."
    )

    st.warning(
        "Recommend academic support sessions."
    )

    if st.button(
    "← Back to Student List",
    key="back_student_list_btn"
):
     st.session_state.page = None

     st.rerun()

# =====================================================
# PREDICTION HISTORY
# =====================================================

def advisor_prediction_history_page():

    st.title(
        "Prediction History"
    )

    st.write(
        "View all prediction history records."
    )

    # =========================================
    # DATABASE
    # =========================================

    conn = sqlite3.connect(
        "users.db"
    )

    try:

        history_df = pd.read_sql_query(

            """

            SELECT

                student_name,
                student_id,
                predicted_risk,
                probability_score,
                semester,
                gpa,
                stress_index,
                prediction_time,
                predicted_by

            FROM prediction_history

            ORDER BY prediction_time DESC

            """,

            conn
        )

    except:

        st.warning(
            "No prediction history available."
        )

        conn.close()

        return

    conn.close()

    # =========================================
    # EMPTY
    # =========================================

    if history_df.empty:

        st.warning(
            "No prediction history found."
        )

        return

    # =========================================
    # FILTER
    # =========================================

    risk_filter = st.selectbox(

        "Filter Risk Level",

        [

            "All",
            "High Risk",
            "Medium Risk",
            "Low Risk"

        ],

        key="prediction_history_filter"

    )

    if risk_filter != "All":

        history_df = history_df[

            history_df["predicted_risk"]

            .astype(str)

            .str.strip()

            == risk_filter

        ]

    # =========================================
    # SEARCH
    # =========================================

    search = st.text_input(

        "Search Student",

        key="prediction_history_search"

    )

    if search:

        history_df = history_df[

            history_df["student_name"]

            .astype(str)

            .str.contains(

                search,

                case=False,

                na=False

            )

        ]

    # =========================================
    # TABLE HEADER
    # =========================================

    st.markdown("---")

    h1, h2, h3, h4, h5, h6, h7 = st.columns(
        [2,2,2,2,2,2,3]
    )

    with h1:
        st.write("Student")

    with h2:
        st.write("Student ID")

    with h3:
        st.write("Risk")

    with h4:
        st.write("Probability")

    with h5:
        st.write("Semester")

    with h6:
        st.write("GPA")

    with h7:
        st.write("Prediction Time")

    st.divider()

    # =========================================
    # ROWS
    # =========================================

    for index, row in history_df.iterrows():

        c1, c2, c3, c4, c5, c6, c7 = st.columns(
            [2,2,2,2,2,2,3]
        )

        with c1:

            st.write(
                str(row["student_name"])
            )

        with c2:

            st.write(
                str(row["student_id"])
            )

        with c3:

            risk = str(
                row["predicted_risk"]
            )

            if risk == "High Risk":

                st.error(risk)

            elif risk == "Medium Risk":

                st.warning(risk)

            else:

                st.success(risk)

        with c4:

            st.write(
                f"{round(float(row['probability_score']),2)}%"
            )

        with c5:

            st.write(
                str(row["semester"])
            )

        with c6:

            st.write(
                round(float(row["gpa"]),2)
            )

        with c7:

            st.write(
                str(row["prediction_time"])
            )

        st.divider()


# =====================================================
# RISK TREND
# =====================================================

def advisor_risk_trend_page():

    st.title(
        "Risk Trend Analysis"
    )

    trend_df = pd.DataFrame({

        "Month": [

            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May"
        ],

        "High Risk": [

            12,
            10,
            9,
            7,
            5
        ]
    })

    st.line_chart(

        trend_df.set_index(
            "Month"
        )
    )



def student_details_page():

    if st.button(
        "← Back to Student List",
        key="back_student_list_btn"
    ):

        st.session_state.page = "student_list"

        st.rerun()

    st.title("Student Details")

    student_id = st.session_state.get(
        "selected_student_id"
    )

    if student_id is None:

        st.warning("No student selected.")

        return

    conn = sqlite3.connect(
        "users.db"
    )

    df = pd.read_sql_query(
        """
        SELECT *
        FROM prediction_history
        WHERE student_id = ?
        ORDER BY prediction_time DESC
        LIMIT 1
        """,
        conn,
        params=(student_id,)
    )

    conn.close()

    if len(df) == 0:

        st.error("Student not found.")

        return

    student = df.iloc[0]

    st.divider()

    col1, col2 = st.columns([1, 2])

    with col1:

        st.image(
            "https://cdn-icons-png.flaticon.com/512/3135/3135715.png",
            width=120
        )

    with col2:

        st.subheader(student["student_name"])

        st.write(
            f"Student ID: {student['student_id']}"
        )

        st.write(
            f"Semester: {student['semester']}"
        )

        st.write(
            f"GPA: {student['gpa']}"
        )

        st.write(
            f"Stress Index: {student['stress_index']}"
        )

        st.write(
            f"Risk Level: {student['predicted_risk']}"
        )

        st.write(
            f"Probability Score: "
            f"{round(student['probability_score'],2)}%"
        )

    st.divider()

    st.subheader("Risk Probability")

    probability = float(
        student["probability_score"]
    ) / 100

    st.progress(probability)

    st.write(
        f"{round(probability * 100,2)}%"
    )

    st.divider()

    st.subheader("Recommended Actions")

    if student["predicted_risk"] == "High Risk":

        st.error(
            "Immediate advisor intervention recommended."
        )

        st.write(
            "- Schedule counselling"
        )

        st.write(
            "- Monitor attendance"
        )

        st.write(
            "- Parent communication"
        )

    elif student["predicted_risk"] == "Medium Risk":

        st.warning(
            "Monitor academic progress closely."
        )

        st.write(
            "- Weekly check-in"
        )

        st.write(
            "- Academic support"
        )

    else:

        st.success(
            "Student currently stable."
        )

        st.write(
            "- Continue regular monitoring"
        )
import streamlit as st
from zhipuai import ZhipuAI


MODEL = "glm-4-flash"

SYSTEM_PROMPT = """
You are AURA, a professional student-at-risk support assistant.
Your role is to help academic staff understand student dropout risk and suggest practical interventions.
Use clear, short English. Do not claim that a prediction is certain. Always frame advice as support guidance.
"""

QUICK_QUESTIONS = [
    "What should I do if a student's GPA is low?",
    "What does high dropout risk mean?",
    "What interventions are suitable for a high-risk student?",
]


def get_client():
    api_key = st.secrets.get("ZHIPUAI_API_KEY", "").strip()
    if not api_key:
        return None
    return ZhipuAI(api_key=api_key)


def chatbot_page():
    st.title("AURA AI Assistant")

    if st.button("Back to Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()

    client = get_client()
    if client is None:
        st.warning("ZHIPUAI_API_KEY is not configured in .streamlit/secrets.toml.")
        st.info("The assistant page is available, but live AI replies are disabled.")
        return

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.write("**Quick Questions:**")
    cols = st.columns(len(QUICK_QUESTIONS))
    for i, question in enumerate(QUICK_QUESTIONS):
        with cols[i]:
            if st.button(question, use_container_width=True, key=f"quick_{i}"):
                st.session_state.chat_history.append(
                    {"role": "user", "content": question}
                )
                st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Type your message...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    *st.session_state.chat_history,
                ],
                stream=True,
            )
            reply = st.write_stream(
                chunk.choices[0].delta.content or "" for chunk in stream
            )

        st.session_state.chat_history.append({"role": "assistant", "content": reply})

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

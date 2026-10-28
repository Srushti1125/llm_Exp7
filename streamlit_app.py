"""Streamlit front end for the Customer Support RAG (Experiment 7)."""
import os

import streamlit as st

st.set_page_config(page_title="Customer Support Assistant", page_icon="💬")

# Make Streamlit secrets visible as environment variables BEFORE importing
# support_rag (it reads its config with os.getenv at import time).
try:
    for key, value in st.secrets.items():
        if isinstance(value, (str, int, float)):
            os.environ.setdefault(key, str(value))
except Exception:
    pass  # no secrets file locally -> fall back to .env / defaults

from support_rag import LLMBackendError, SupportRAG, LLM_BACKEND  # noqa: E402

MAX_QUESTION_LENGTH = 1000


@st.cache_resource(show_spinner="Loading support FAQ...")
def load_rag() -> SupportRAG:
    # Built once per server process, not on every rerun.
    return SupportRAG()


st.title("💬 Customer Support Assistant")
st.caption(f"Answers come only from the support FAQ · backend: `{LLM_BACKEND}`")

rag = load_rag()

if "history" not in st.session_state:
    st.session_state.history = []

for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn.get("meta"):
            st.caption(turn["meta"])

question = st.chat_input("Ask a support question (orders, refunds, account, billing...)")

if question:
    question = question.strip()
    if not question:
        st.stop()
    if len(question) > MAX_QUESTION_LENGTH:
        st.warning(f"Please keep your question under {MAX_QUESTION_LENGTH} characters.")
        st.stop()

    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = rag.ask(question)
                answer = result["answer"]
                meta = (
                    f"intent: {result['intent']} · retries: {result['retries']} · "
                    f"fallback: {result['fallback_used']}"
                )
            except LLMBackendError:
                answer = "The language model service is unavailable. Please try again later."
                meta = ""
            except Exception:
                answer = "Something went wrong. Please try again."
                meta = ""
        st.markdown(answer)
        if meta:
            st.caption(meta)

    st.session_state.history.append({"role": "assistant", "content": answer, "meta": meta})

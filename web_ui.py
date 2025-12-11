"""Streamlit UI for chatting with the RAG backend."""
from __future__ import annotations

import os
from typing import Any, Dict, List

import requests
import streamlit as st


API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")


def post_chat(question: str) -> str:
    response = requests.post(f"{API_URL}/chat", json={"question": question}, timeout=60)
    response.raise_for_status()
    return response.json().get("answer", "")


def upload_file(uploaded) -> Dict[str, Any]:
    files = {"file": (uploaded.name, uploaded, uploaded.type)}
    response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
    response.raise_for_status()
    return response.json()


with st.sidebar:
    st.header("Knowledge Base")
    uploaded = st.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
    if uploaded and st.button("Ingest document"):
        with st.spinner("Processing document..."):
            try:
                info = upload_file(uploaded)
                st.success(
                    f"Uploaded {info.get('filename')} | chunks added: {info.get('chunks_added')}"
                )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Upload failed: {exc}")


if "messages" not in st.session_state:
    st.session_state.messages: List[Dict[str, str]] = []

# Render history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your documents"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching and composing..."):
            try:
                answer = post_chat(prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Request failed: {exc}")

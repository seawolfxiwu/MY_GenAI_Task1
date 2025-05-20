import os
from typing import Dict, List, Tuple
import streamlit as st

def load_utterances(files: List) -> Tuple[Dict, Dict]:
    """Load and separate member and agent utterances from uploaded files"""
    documents = {}  # Member-Only Utterances
    agent_docs = {}  # Agent-Only Utterances

    for file in files:
        try:
            file_name = os.path.splitext(file.name)[0]
            file_ext = os.path.splitext(file.name)[-1].lower()

            if file_ext == ".txt":
                text = file.read().decode("utf-8")
                if text:
                    lines = text.split("\n")

                    # Process member lines
                    member_lines = [
                        line.split("Member:", 1)[1].strip()
                        for line in lines
                        if line.startswith("Member:")
                    ]
                    documents[file_name] = "\n".join(member_lines)

                    # Process agent lines
                    agent_lines = [
                        line.split("PA Agent:", 1)[1].strip()
                        for line in lines
                        if line.startswith("PA Agent:")
                    ]
                    agent_docs[file_name] = "\n".join(agent_lines)
        except Exception as e:
            st.error(f"Error processing file {file.name}: {str(e)}")

    return documents, agent_docs

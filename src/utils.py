from typing import Dict, List
import streamlit as st

def clean_text(text: str, max_length: int = 128) -> str:
    """Clean and truncate text for display"""
    text = text.strip()
    if len(text) > max_length:
        return text[:max_length] + '...'
    return text

def validate_prompts(sys_instructions: List[str], prompts: List[str]) -> bool:
    """Validate that system instructions and prompts are properly mapped"""
    return len(sys_instructions) == len(prompts)

def initialize_session_state():
    """Initialize session state variables if they don't exist"""
    if 'documents' not in st.session_state:
        st.session_state.documents = {}
    if 'agent_docs' not in st.session_state:
        st.session_state.agent_docs = {}
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

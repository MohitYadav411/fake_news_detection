"""
pages/5_System_Logs.py
FakeNewsAI - System Logs & Audit Trail Page.
"""

import streamlit as st
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ui_theme import inject_theme
from views import render_system_logs_page

st.set_page_config(
    page_title="FakeNewsAI - System Logs",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme()

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 25px;">
            <span style="font-size: 2rem;">🛡️</span>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">FakeNewsAI</h3>
                <div style="font-size: 0.8rem; color: #94A3B8;">System Logs</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if hasattr(st, "page_link"):
        st.page_link("app.py", label="🏠 Main Application", icon=None)
        st.page_link("pages/admin.py", label="⚙️ Admin Dashboard", icon=None)
    else:
        if st.button("🏠 Main Application", use_container_width=True):
            st.switch_page("app.py")
        if st.button("⚙️ Admin Dashboard", use_container_width=True):
            st.switch_page("pages/admin.py")

render_system_logs_page()

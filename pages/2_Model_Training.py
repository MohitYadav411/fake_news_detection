"""
pages/2_Model_Training.py
FakeNewsAI - Model Training & Versioning Page.
"""

import streamlit as st
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from ui_theme import inject_theme
from views import render_model_training_page

st.set_page_config(
    page_title="FakeNewsAI - Model Training",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_theme()

with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 25px;">
            <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%); border: 1px solid rgba(99, 102, 241, 0.5); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; color: #818CF8; box-shadow: 0 0 16px rgba(99, 102, 241, 0.35);">
                <i class="fa-solid fa-shield-halved"></i>
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">FakeNewsAI</h3>
                <div style="font-size: 0.8rem; color: #94A3B8;">Model Training</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    if hasattr(st, "page_link"):
        st.page_link("app.py", label="Main Application", icon=":material/home:")
        st.page_link("pages/admin.py", label="Admin Dashboard", icon=":material/admin_panel_settings:")
    else:
        if st.button("Main Application", icon=":material/home:", use_container_width=True):
            st.switch_page("app.py")
        if st.button("Admin Dashboard", icon=":material/admin_panel_settings:", use_container_width=True):
            st.switch_page("pages/admin.py")

render_model_training_page(is_admin=True)

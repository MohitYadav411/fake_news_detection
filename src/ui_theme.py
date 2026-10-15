"""
src/ui_theme.py
Centralized CSS styling and UI components for FakeNewsAI.
Matches the modern dark-themed aesthetics shown in the design specification.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
/* Global Dark Theme Settings */
:root {
    --bg-main: #0B1120;
    --bg-card: #0F172A;
    --bg-surface: #1E293B;
    --border-color: #334155;
    --border-light: #1E293B;
    --accent-blue: #2563EB;
    --accent-blue-hover: #1D4ED8;
    --accent-cyan: #0284C7;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
    --text-muted: #64748B;
    --success-green: #10B981;
    --danger-red: #EF4444;
    --warning-yellow: #F59E0B;
}

/* Base Body and Streamlit container */
.stApp {
    background-color: var(--bg-main) !important;
    color: var(--text-primary) !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #070D18 !important;
    border-right: 1px solid #1E293B !important;
}

/* Hide default multipage navigation list from sidebar */
[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stSidebar"] .stMarkdown h1, 
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #F8FAFC !important;
}

/* Primary Button Styling */
.stButton > button {
    background-color: var(--accent-blue) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease-in-out !important;
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
}

.stButton > button:hover {
    background-color: var(--accent-blue-hover) !important;
    box-shadow: 0 6px 16px rgba(37, 99, 235, 0.35) !important;
    transform: translateY(-1px) !important;
}

/* Text Inputs and Text Areas */
.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 8px !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: var(--accent-blue) !important;
    box-shadow: 0 0 0 1px var(--accent-blue) !important;
}

/* Hide default streamlit header elements slightly for cleaner look */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Card Container */
.fna-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1.25rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
}

.fna-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.8rem;
    border-bottom: 1px solid #1E293B;
    padding-bottom: 0.5rem;
}

.fna-card-title {
    font-size: 1.05rem;
    font-weight: 600;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Status Badges */
.fna-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.25rem 0.65rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}

.badge-real {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-fake {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-unverified {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-verified {
    background-color: rgba(14, 165, 233, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(14, 165, 233, 0.3);
}

.badge-reviewed {
    background-color: rgba(59, 130, 246, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(59, 130, 246, 0.3);
}

.badge-pending {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-active {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-inactive {
    background-color: rgba(100, 116, 139, 0.15);
    color: #94A3B8;
    border: 1px solid rgba(100, 116, 139, 0.3);
}

.badge-source {
    background-color: rgba(37, 99, 235, 0.15);
    color: #60A5FA;
    border: 1px solid rgba(37, 99, 235, 0.3);
}

.badge-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-warning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

.badge-error {
    background-color: rgba(239, 68, 68, 0.15);
    color: #F87171;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

.badge-info {
    background-color: rgba(14, 165, 233, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(14, 165, 233, 0.3);
}

/* Top App Header & Pill */
.fna-top-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1E293B;
}

.fna-pill-ai {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background-color: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 0.4rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    color: #34D399;
}

/* KPI Card Grid */
.kpi-container {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 500;
    color: #94A3B8;
    margin-bottom: 0.25rem;
}

.kpi-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #F8FAFC;
    line-height: 1.2;
}

.kpi-subtext {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.4rem;
}

.kpi-bar {
    height: 4px;
    border-radius: 2px;
    margin-top: 0.5rem;
    width: 100%;
}

/* Empty State Card */
.fna-empty-state {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 3rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
}

.fna-empty-icon {
    font-size: 2.8rem;
    color: #3B82F6;
    margin-bottom: 0.75rem;
    display: inline-block;
    padding: 0.75rem;
    border-radius: 12px;
    background-color: rgba(37, 99, 235, 0.1);
    border: 1px solid rgba(37, 99, 235, 0.25);
}

/* Prediction Details side panel */
.detail-panel {
    background-color: #0F172A;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.25rem;
}
</style>
"""

def inject_theme():
    """Injects the custom CSS into the Streamlit session."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def render_kpi(title, value, subtext=None, bar_color=None, icon=None):
    """Renders a styled dark KPI card."""
    icon_html = f"<span style='margin-right: 6px;'>{icon}</span>" if icon else ""
    bar_html = f"<div class='kpi-bar' style='background-color: {bar_color};'></div>" if bar_color else ""
    sub_html = f"<div class='kpi-subtext'>{subtext}</div>" if subtext else ""
    
    html = f"""
    <div class="kpi-container">
        <div>
            <div class="kpi-title">{icon_html}{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        <div>
            {bar_html}
            {sub_html}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_badge(label, badge_type="real"):
    """Returns HTML for a rounded status pill badge."""
    css_class = f"badge-{badge_type.lower()}"
    return f"<span class='fna-badge {css_class}'>{label}</span>"

def render_stat_card(title: str, value: str, subtext: str = "", icon: str = "", color: str = "#3B82F6"):
    """Renders a styled stat card with icon on dark background matching Screenshot 1."""
    html = f"""
    <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 1.1rem; display: flex; align-items: center; gap: 14px; height: 100%;">
        <div style="background-color: rgba(255, 255, 255, 0.05); border: 1px solid #1E293B; border-radius: 10px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; color: {color}; flex-shrink: 0;">
            {icon}
        </div>
        <div>
            <div style="font-size: 0.78rem; color: #94A3B8; font-weight: 500;">{title}</div>
            <div style="font-size: 1.35rem; font-weight: 700; color: #F8FAFC; line-height: 1.25;">{value}</div>
            <div style="font-size: 0.72rem; color: #64748B; margin-top: 2px;">{subtext}</div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

def render_confidence_pill(confidence: float, label: str):
    """Renders a numeric confidence with colored mini progress bar."""
    color = "#10B981" if str(label).lower() == "real" else ("#EF4444" if str(label).lower() == "fake" else "#F59E0B")
    pct = min(max(float(confidence), 0.0), 100.0)
    html = f"""
    <div style="display: flex; align-items: center; gap: 8px;">
        <span style="font-size: 0.82rem; font-weight: 600; color: #CBD5E1; min-width: 42px;">{confidence}%</span>
        <div style="background-color: #1E293B; border-radius: 999px; height: 6px; width: 50px; overflow: hidden;">
            <div style="background-color: {color}; height: 100%; width: {pct}%;"></div>
        </div>
    </div>
    """
    return html


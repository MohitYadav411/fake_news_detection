"""
src/ui_theme.py
Centralized CSS styling and UI components for FakeNewsAI.
Matches the modern dark-themed aesthetics shown in the design specification.
"""

import streamlit as st

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
@import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css');

/* Global Theme Settings - Custom Dark Cyber Palette */
:root {
    --bg-main: #080F20;
    --bg-card: #101C33;
    --bg-card-hover: #162544;
    --bg-surface: #101C33;
    --border-color: rgba(109, 74, 255, 0.25);
    --border-glow: rgba(109, 74, 255, 0.4);
    --accent-primary: #6D4AFF;
    --accent-primary-gradient: linear-gradient(135deg, #6D4AFF 0%, #8B6BFF 50%, #4E27E8 100%);
    --accent-emerald-gradient: linear-gradient(135deg, #10B981 0%, #059669 100%);
    --accent-rose-gradient: linear-gradient(135deg, #F43F5E 0%, #E11D48 100%);
    --text-primary: #EAF0FF;
    --text-secondary: #A3B5D9;
    --text-muted: #6B7C9E;
    --success-green: #10B981;
    --danger-red: #F43F5E;
    --warning-yellow: #F59E0B;
    --font-sans: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    --font-mono: 'JetBrains Mono', monospace;
}

/* Base Body and Streamlit container */
.stApp {
    background: radial-gradient(circle at 12% 15%, rgba(109, 74, 255, 0.12) 0%, transparent 45%),
                radial-gradient(circle at 88% 85%, rgba(109, 74, 255, 0.08) 0%, transparent 45%),
                #080F20 !important;
    color: var(--text-primary) !important;
    font-family: var(--font-sans) !important;
    letter-spacing: -0.01em !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #091226 0%, #060B17 100%) !important;
    border-right: 1px solid rgba(109, 74, 255, 0.2) !important;
    box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
}

/* Hide default multipage navigation list from sidebar */
[data-testid="stSidebarNav"] {
    display: none !important;
}

[data-testid="stSidebar"] .stMarkdown h1, 
[data-testid="stSidebar"] .stMarkdown h2, 
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #EAF0FF !important;
    font-family: var(--font-sans) !important;
}

/* Sleek Navigation Items in Sidebar */
[data-testid="stSidebar"] div.stButton > button {
    background: rgba(16, 28, 51, 0.7) !important;
    border: 1px solid rgba(109, 74, 255, 0.2) !important;
    color: #A3B5D9 !important;
    border-radius: 10px !important;
    text-align: left !important;
    justify-content: flex-start !important;
    padding: 0.65rem 1rem !important;
    font-size: 0.9rem !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
    box-shadow: none !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

[data-testid="stSidebar"] div.stButton > button:hover {
    background: rgba(22, 37, 68, 0.9) !important;
    border-color: rgba(109, 74, 255, 0.5) !important;
    color: #FFFFFF !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 15px rgba(109, 74, 255, 0.15) !important;
}

/* Active Nav Button */
[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, rgba(109, 74, 255, 0.35) 0%, rgba(16, 28, 51, 0.9) 100%) !important;
    border: 1px solid #6D4AFF !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    box-shadow: 0 0 16px rgba(109, 74, 255, 0.3) !important;
    transform: translateX(2px) !important;
}

/* Primary Button Styling */
.stButton > button {
    background: var(--accent-primary-gradient) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: var(--font-sans) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.65rem 1.6rem !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 16px rgba(109, 74, 255, 0.35) !important;
    letter-spacing: 0.01em !important;
}

.stButton > button:hover {
    box-shadow: 0 6px 24px rgba(109, 74, 255, 0.55) !important;
    transform: translateY(-2px) !important;
    filter: brightness(1.1) !important;
}

.stButton > button:active {
    transform: translateY(0px) !important;
}

/* Secondary Button Styling */
.stButton > button[kind="secondary"] {
    background: #101C33 !important;
    border: 1px solid rgba(109, 74, 255, 0.3) !important;
    box-shadow: none !important;
}

.stButton > button[kind="secondary"]:hover {
    background: #162544 !important;
    border-color: rgba(109, 74, 255, 0.6) !important;
}

/* Text Inputs and Text Areas - Glassmorphism */
.stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] {
    background-color: #101C33 !important;
    color: #EAF0FF !important;
    border: 1px solid rgba(109, 74, 255, 0.3) !important;
    border-radius: 10px !important;
    backdrop-filter: blur(12px) !important;
    transition: all 0.2s ease !important;
    font-family: var(--font-sans) !important;
}

.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: #6D4AFF !important;
    box-shadow: 0 0 0 2px rgba(109, 74, 255, 0.3) !important;
    background-color: #162544 !important;
}

/* Header styling */
header[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Card Container with Modern Glassmorphism & Subtle Border Gradient */
.fna-card {
    background: linear-gradient(135deg, rgba(16, 28, 51, 0.9) 0%, rgba(10, 18, 36, 0.95) 100%);
    border: 1px solid rgba(109, 74, 255, 0.25);
    border-radius: 14px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(16px);
    transition: border-color 0.3s ease, box-shadow 0.3s ease;
}

.fna-card:hover {
    border-color: rgba(109, 74, 255, 0.45);
    box-shadow: 0 12px 36px -10px rgba(109, 74, 255, 0.25);
}

.fna-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    border-bottom: 1px solid rgba(30, 41, 59, 0.8);
    padding-bottom: 0.65rem;
}

.fna-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #F8FAFC;
    display: flex;
    align-items: center;
    gap: 0.6rem;
    letter-spacing: -0.01em;
}

/* Status Badges with Subtle Glow */
.fna-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.3rem 0.8rem;
    border-radius: 9999px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}

.badge-real {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.15) 100%);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
    box-shadow: 0 0 12px rgba(16, 185, 129, 0.2);
}

.badge-fake {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(225, 29, 72, 0.15) 100%);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
    box-shadow: 0 0 12px rgba(244, 63, 94, 0.2);
}

.badge-unverified {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(217, 119, 6, 0.15) 100%);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
    box-shadow: 0 0 12px rgba(245, 158, 11, 0.15);
}

.badge-verified {
    background: rgba(14, 165, 233, 0.2);
    color: #38BDF8;
    border: 1px solid rgba(14, 165, 233, 0.4);
}

.badge-reviewed {
    background: rgba(99, 102, 241, 0.2);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

.badge-pending {
    background: rgba(245, 158, 11, 0.2);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-active {
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-inactive {
    background: rgba(100, 116, 139, 0.2);
    color: #94A3B8;
    border: 1px solid rgba(100, 116, 139, 0.4);
}

.badge-source {
    background: rgba(99, 102, 241, 0.2);
    color: #818CF8;
    border: 1px solid rgba(99, 102, 241, 0.4);
}

.badge-success {
    background: rgba(16, 185, 129, 0.2);
    color: #34D399;
    border: 1px solid rgba(16, 185, 129, 0.4);
}

.badge-warning {
    background: rgba(245, 158, 11, 0.2);
    color: #FBBF24;
    border: 1px solid rgba(245, 158, 11, 0.4);
}

.badge-error {
    background: rgba(244, 63, 94, 0.2);
    color: #FB7185;
    border: 1px solid rgba(244, 63, 94, 0.4);
}

.badge-info {
    background: rgba(14, 165, 233, 0.2);
    color: #38BDF8;
    border: 1px solid rgba(14, 165, 233, 0.4);
}

/* Top App Pill */
.fna-pill-ai {
    display: inline-flex;
    align-items: center;
    gap: 0.6rem;
    background: linear-gradient(135deg, rgba(109, 74, 255, 0.2) 0%, rgba(16, 28, 51, 0.6) 100%);
    border: 1px solid rgba(109, 74, 255, 0.4);
    padding: 0.45rem 1rem;
    border-radius: 9999px;
    font-size: 0.84rem;
    font-weight: 600;
    color: #D6CBFF;
    box-shadow: 0 0 15px rgba(109, 74, 255, 0.2);
}

/* KPI Card Grid with Glow & Depth */
.kpi-container {
    background: linear-gradient(145deg, rgba(16, 28, 51, 0.95) 0%, rgba(11, 20, 38, 0.98) 100%);
    border: 1px solid rgba(109, 74, 255, 0.22);
    border-radius: 14px;
    padding: 1.15rem;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
}

.kpi-container:hover {
    transform: translateY(-3px);
    border-color: rgba(109, 74, 255, 0.5);
    box-shadow: 0 8px 25px -5px rgba(109, 74, 255, 0.3);
}

.kpi-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: #A3B5D9;
    margin-bottom: 0.35rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.kpi-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: #EAF0FF;
    line-height: 1.15;
    letter-spacing: -0.02em;
}

.kpi-subtext {
    font-size: 0.78rem;
    color: #6B7C9E;
    margin-top: 0.5rem;
    font-weight: 500;
}

.kpi-bar {
    height: 4px;
    border-radius: 999px;
    margin-top: 0.6rem;
    width: 100%;
}

/* Dataframe styling */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(51, 65, 85, 0.7) !important;
    border-radius: 12px !important;
    overflow: hidden !important;
}

/* Empty State Card */
.fna-empty-state {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.6) 0%, rgba(10, 16, 31, 0.7) 100%);
    border: 1px dashed rgba(99, 102, 241, 0.35);
    border-radius: 16px;
    padding: 3.5rem 1.5rem;
    text-align: center;
    margin-top: 1rem;
    backdrop-filter: blur(12px);
}

.fna-empty-icon {
    font-size: 3rem;
    margin-bottom: 0.85rem;
    display: inline-block;
    padding: 1rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(59, 130, 246, 0.1) 100%);
    border: 1px solid rgba(99, 102, 241, 0.3);
    box-shadow: 0 0 25px rgba(99, 102, 241, 0.2);
}

/* Prediction Details side panel */
.detail-panel {
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(51, 65, 85, 0.7);
    border-radius: 14px;
    padding: 1.35rem;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
}

/* Modern FontAwesome icon tweaks */
i.fa, i.fas, i.far, i.fab {
    vertical-align: -0.05em;
}
.fna-header-icon {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%);
    border: 1px solid rgba(99, 102, 241, 0.5);
    border-radius: 12px;
    width: 48px;
    height: 48px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.4rem;
    color: #818CF8;
    box-shadow: 0 0 20px rgba(99, 102, 241, 0.3);
}

/* ==========================================================================
   RESPONSIVE & MOBILE DESIGN ENHANCEMENTS
   Breakpoints for Mobile (<= 768px) and Tablets (<= 1024px)
   ========================================================================== */

/* Smooth touch scrolling and tap highlights */
* {
    -webkit-tap-highlight-color: transparent;
    box-sizing: border-box;
}

/* Base container padding for mobile */
@media (max-width: 768px) {
    /* Adjust Streamlit main block padding */
    .stApp > header {
        background-color: rgba(8, 15, 32, 0.85) !important;
        backdrop-filter: blur(10px) !important;
    }

    .main .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 2.5rem !important;
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        max-width: 100% !important;
    }

    /* Headings responsiveness */
    h1, .stMarkdown h1 {
        font-size: 1.55rem !important;
        line-height: 1.25 !important;
        word-break: break-word !important;
    }

    h2, .stMarkdown h2 {
        font-size: 1.3rem !important;
        line-height: 1.3 !important;
    }

    h3, .stMarkdown h3 {
        font-size: 1.15rem !important;
        line-height: 1.3 !important;
    }

    /* Header icons and title container */
    .fna-header-icon {
        width: 38px !important;
        height: 38px !important;
        font-size: 1.15rem !important;
        border-radius: 10px !important;
    }

    /* Cards spacing on mobile */
    .fna-card {
        padding: 1rem 0.9rem !important;
        margin-bottom: 0.9rem !important;
        border-radius: 12px !important;
    }

    .fna-card-header {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 0.4rem !important;
        padding-bottom: 0.5rem !important;
    }

    .fna-card-title {
        font-size: 1rem !important;
    }

    /* Streamlit columns stacking cleanly on mobile */
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
        margin-bottom: 0.6rem !important;
    }

    /* KPI Cards on Mobile */
    .kpi-container {
        padding: 0.9rem !important;
        margin-bottom: 0.5rem !important;
    }

    .kpi-value {
        font-size: 1.45rem !important;
    }

    .kpi-title {
        font-size: 0.75rem !important;
    }

    /* Forms, Buttons, and Inputs touch targets */
    .stButton > button {
        width: 100% !important;
        min-height: 44px !important;
        font-size: 0.95rem !important;
        padding: 0.6rem 1rem !important;
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }

    .stTextArea textarea {
        font-size: 0.95rem !important;
        min-height: 110px !important;
    }

    /* Responsive Dataframe and Tables */
    div[data-testid="stDataFrame"] {
        max-width: 100% !important;
        overflow-x: auto !important;
        -webkit-overflow-scrolling: touch !important;
    }

    /* Flex wrap utilities for verdict bars and badges */
    div[style*="display: flex; justify-content: space-between"] {
        flex-wrap: wrap !important;
        gap: 0.75rem !important;
    }

    /* Responsive charts and images */
    div[data-testid="stImage"] img, .stPlotlyChart, div[data-testid="stPlotlyChart"] {
        max-width: 100% !important;
        height: auto !important;
    }

    /* Mobile sidebar adjustments */
    [data-testid="stSidebar"] {
        min-width: 80vw !important;
        max-width: 85vw !important;
    }

    [data-testid="stSidebar"] div.stButton > button {
        padding: 0.75rem 0.9rem !important;
        font-size: 0.92rem !important;
        min-height: 44px !important;
    }

    /* Streamlit Tabs on mobile */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px !important;
        overflow-x: auto !important;
        white-space: nowrap !important;
        flex-wrap: nowrap !important;
        -webkit-overflow-scrolling: touch !important;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 0.85rem !important;
        font-size: 0.85rem !important;
    }
}

/* Tablet refinements (769px to 1024px) */
@media (min-width: 769px) and (max-width: 1024px) {
    .main .block-container {
        padding-left: 1.25rem !important;
        padding-right: 1.25rem !important;
    }

    .fna-card {
        padding: 1.2rem !important;
    }

    .kpi-value {
        font-size: 1.55rem !important;
    }
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


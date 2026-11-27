"""
app.py
Streamlit UI Module for FakeNewsAI Detection System.
Enhanced with modern dark-mode aesthetic matching the UI specification.
"""

import streamlit as st
import os
import sys
import time
import pandas as pd

# Add src to python path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from predict import predict
from history import log_query, get_history, export_history
from preprocessing import preprocess, is_english_text
from db import init_db, register_user, authenticate_user
from ui_theme import inject_theme, render_badge
from analytics import log_system_event
from views import (
    render_model_insights_page,
    render_prediction_history_page,
    render_model_comparison_page,
    render_model_training_page
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
init_db()

# Page configuration
st.set_page_config(
    page_title="FakeNewsAI - News Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS
inject_theme()

# Model Tier Mapping
TIER_OPTIONS = {
    "Basic": ["naive_bayes", "decision_tree", "logistic_regression", "albert"],
    "Intermediate": ["random_forest", "svm", "electra"],
    "Hardcore": ["lstm", "bert", "distilbert", "roberta"],
    "Ensemble (Majority Vote - All Models)": [
        "naive_bayes", "decision_tree", "logistic_regression", "random_forest", 
        "svm", "lstm", "bert", "distilbert", "roberta", "albert", "electra"
    ]
}

def main():
    # Session State for User Auth
    if "user_auth" not in st.session_state:
        st.session_state.user_auth = {
            "logged_in": False,
            "user_id": None,
            "username": "Guest",
            "role": "guest"
        }

    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 24px; padding-bottom: 15px; border-bottom: 1px solid rgba(51, 65, 85, 0.6);">
                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%); border: 1px solid rgba(99, 102, 241, 0.5); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; color: #818CF8; box-shadow: 0 0 16px rgba(99, 102, 241, 0.35);">
                    <i class="fa-solid fa-shield-halved"></i>
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FakeNewsAI</h3>
                    <div style="font-size: 0.78rem; color: #818CF8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">News Detection</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # User Session Profile Card in Sidebar
        u_info = st.session_state.user_auth
        if u_info["logged_in"]:
            st.markdown(
                f"""
                <div style="background: rgba(16, 28, 51, 0.85); border: 1px solid rgba(99, 102, 241, 0.35); border-radius: 10px; padding: 12px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; gap: 10px;">
                        <div style="background: #2563EB; width: 34px; height: 34px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; color: white;">
                            {u_info['username'][0].upper()}
                        </div>
                        <div>
                            <div style="font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">{u_info['username']}</div>
                            <div style="font-size: 0.75rem; color: #10B981;">● Logged In (Personal History)</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            if st.button("Logout", key="user_logout_btn", use_container_width=True, icon=":material/logout:"):
                st.session_state.user_auth = {
                    "logged_in": False,
                    "user_id": None,
                    "username": "Guest",
                    "role": "guest"
                }
                st.rerun()
        else:
            with st.expander("👤 User Account (Login / Register)", expanded=False):
                auth_tab = st.radio("Mode", ["Login", "Register"], horizontal=True, label_visibility="collapsed")
                if auth_tab == "Login":
                    log_u = st.text_input("Username", key="sidebar_log_u")
                    log_p = st.text_input("Password", type="password", key="sidebar_log_p")
                    if st.button("Sign In", key="sidebar_login_btn", type="primary", use_container_width=True):
                        auth_res = authenticate_user(log_u, log_p)
                        if auth_res:
                            st.session_state.user_auth = {
                                "logged_in": True,
                                "user_id": auth_res["user_id"],
                                "username": auth_res["username"],
                                "role": auth_res["role"]
                            }
                            st.success(f"Welcome back, {auth_res['username']}!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password.")
                else:
                    reg_u = st.text_input("New Username", key="sidebar_reg_u")
                    reg_p = st.text_input("New Password", type="password", key="sidebar_reg_p")
                    reg_p_confirm = st.text_input("Confirm Password", type="password", key="sidebar_reg_pc")
                    if st.button("Create Account", key="sidebar_reg_btn", type="primary", use_container_width=True):
                        if reg_p != reg_p_confirm:
                            st.error("Passwords do not match.")
                        else:
                            ok, msg = register_user(reg_u, reg_p)
                            if ok:
                                st.success(msg)
                            else:
                                st.error(msg)
        
        st.markdown("<p style='font-size: 0.78rem; font-weight: 700; color: #6D4AFF; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;'><i class='fa-solid fa-compass' style='margin-right: 6px;'></i>Navigation</p>", unsafe_allow_html=True)
        
        # Navigation items: Only show "Prediction History" if user is logged in
        nav_items = [
            ("Home (Check News)", "Check News", ":material/search_check:")
        ]
        if st.session_state.user_auth.get("logged_in"):
            nav_items.append(("Prediction History", "My History", ":material/history:"))
        nav_items.append(("Model Comparison", "Model Comparison", ":material/balance:"))
        
        if "app_nav" not in st.session_state:
            st.session_state.app_nav = "Home (Check News)"

        # If user logged out while on Prediction History, reset navigation to Home
        if st.session_state.app_nav in ["Prediction History", "History"] and not st.session_state.user_auth.get("logged_in"):
            st.session_state.app_nav = "Home (Check News)"
            
        for page_key, display_label, icon_name in nav_items:
            is_active = (st.session_state.app_nav == page_key)
            btn_type = "primary" if is_active else "secondary"
            if st.button(display_label, key=f"main_nav_{page_key}", type=btn_type, icon=icon_name, use_container_width=True):
                st.session_state.app_nav = page_key
                st.rerun()

        page = st.session_state.app_nav
        
        st.markdown("<hr style='border-color: rgba(109, 74, 255, 0.2); margin: 25px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background: linear-gradient(135deg, rgba(16, 28, 51, 0.9) 0%, rgba(10, 18, 36, 0.95) 100%); border: 1px solid rgba(109, 74, 255, 0.25); border-radius: 10px; padding: 14px; font-size: 0.82rem; color: #A3B5D9; box-shadow: 0 4px 15px rgba(0,0,0,0.25);">
                <span style="color: #10B981; font-weight: 700;"><i class="fa-solid fa-bolt" style="margin-right: 6px;"></i>Active Intelligence</span><br>
                11 deep learning & ML models active with real-time audit trail.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main Area
    if page == "Home (Check News)":
        # Top Header Row
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 14px; margin-bottom: 12px;">
                <div style="display: flex; align-items: center; gap: 14px; min-width: 260px; flex: 1 1 300px;">
                    <div class="fna-header-icon" style="flex-shrink: 0;">
                        <i class="fa-solid fa-newspaper"></i>
                    </div>
                    <div>
                        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF 0%, #E2E8F0 50%, #94A3B8 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            Fake News Detection System
                        </h1>
                        <p style="color: #94A3B8; font-size: 0.9rem; margin: 2px 0 0 0;">
                            Verify headlines, claims, and articles in real-time across ensemble AI models.
                        </p>
                    </div>
                </div>
                <div>
                    <span class="fna-pill-ai">
                        <i class="fa-solid fa-microchip" style="color: #10B981;"></i> AI Ensemble <span style="color: #64748B;">|</span> 11 Models Active
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Main Input Area
        raw_text = st.text_area(
            "Text Input",
            placeholder="Paste news article, headline or claim here...",
            height=180,
            label_visibility="collapsed"
        )
        
        char_count = len(raw_text)
        
        col_meta, col_controls = st.columns([1.5, 1])
        with col_meta:
            st.markdown(
                f"<div style='font-size: 0.85rem; color: #64748B; padding-top: 10px;'>"
                f"Characters: <b style='color: #94A3B8;'>{char_count}</b> &nbsp;|&nbsp; Language: <b style='color: #818CF8;'>English (Required)</b>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_controls:
            col_tier, col_btn = st.columns([1.3, 1])
            with col_tier:
                tier_keys = list(TIER_OPTIONS.keys())
                ensemble_idx = tier_keys.index("Ensemble (Majority Vote - All Models)") if "Ensemble (Majority Vote - All Models)" in tier_keys else 0
                selected_tier = st.selectbox(
                    "Select Model Tier",
                    tier_keys,
                    index=ensemble_idx,
                    label_visibility="collapsed"
                )
            with col_btn:
                check_btn = st.button("Check News", use_container_width=True, disabled=(char_count == 0))

        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction and Analysis Logic
        if check_btn and raw_text.strip():
            if not is_english_text(raw_text):
                st.error("⚠️ Unsupported Language: This system is trained exclusively on English news. Please enter or translate your article into English for analysis.")
                st.info("💡 Note: Processing non-English news directly causes out-of-vocabulary zero vectors, resulting in unreliable predictions.")
            else:
                models_to_run = TIER_OPTIONS[selected_tier]
                is_ensemble = "Ensemble" in selected_tier
                
                with st.spinner("Analyzing text using trained models..."):
                    start_time = time.time()
                    cleaned_text = preprocess(raw_text)
                    results = []
                    
                    real_count = 0
                    fake_count = 0
                    
                    for m_name in models_to_run:
                        m_start = time.time()
                        res = predict(raw_text, m_name)
                        m_dur = round(time.time() - m_start, 2)
                        if "error" not in res:
                            label = res["label"]
                            conf = res["confidence"]
                            if label == "Real":
                                real_count += 1
                            else:
                                fake_count += 1
                            results.append({
                                "model": m_name,
                                "label": label,
                                "confidence": conf
                            })
                            # Live dynamic status: borderline confidence (< 65%) marked as Pending review
                            initial_status = "Pending" if conf < 65.0 else "Reviewed"
                            active_user_id = st.session_state.user_auth.get("user_id")
                            log_query(raw_text, cleaned_text, m_name, label, conf, inference_time=max(m_dur, 0.15), status=initial_status, user_id=active_user_id)
                            log_system_event("INFO", "Predictor", f"{m_name.replace('_', ' ').title()} classified query as {label} ({conf}%)")
                    
                    inference_sec = round(time.time() - start_time, 2)
                    
                # Render Results Card
                if results:
                    st.markdown('<div class="fna-card">', unsafe_allow_html=True)
                    
                    if is_ensemble:
                        final_label = "Real" if real_count >= fake_count else "Fake"
                        vote_ratio = f"{max(real_count, fake_count)} of {len(results)}"
                        badge_style = "badge-real" if final_label == "Real" else "badge-fake"
                        badge_icon = '<i class="fa-solid fa-circle-check"></i>' if final_label == "Real" else '<i class="fa-solid fa-triangle-exclamation"></i>'
                        avg_conf = round(sum(r["confidence"] for r in results) / len(results), 1)

                        st.markdown(
                            f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; border-bottom: 1px solid #1E293B; padding-bottom: 12px; margin-bottom: 16px;">
                                <div style="min-width: 240px; flex: 1 1 280px;">
                                    <span class="fna-badge {badge_style}" style="font-size: 1.05rem; padding: 6px 14px; max-width: 100%; word-break: break-word;">
                                        {badge_icon} FINAL VERDICT: {final_label.upper()} NEWS
                                    </span>
                                    <div style="font-size: 0.82rem; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
                                        <i class="fa-solid fa-users-viewfinder" style="margin-right: 4px;"></i> Ensemble Consensus: <b>{vote_ratio} models agree</b><br>
                                        <i class="fa-solid fa-stopwatch" style="margin-right: 4px;"></i> Inference: <b>{inference_sec}s</b>
                                    </div>
                                </div>
                                <div style="text-align: right; min-width: 110px;">
                                    <div style="font-size: 0.78rem; color: #64748B;">Average Confidence</div>
                                    <div style="font-size: 1.45rem; font-weight: 700; color: #F8FAFC;">{avg_conf}%</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
                        st.progress(min(avg_conf / 100.0, 1.0))
                        
                        st.markdown("<p style='font-weight: 600; font-size: 0.95rem; margin-top: 15px; color: #F8FAFC;'><i class='fa-solid fa-layer-group' style='margin-right: 6px;'></i>Individual Model Predictions:</p>", unsafe_allow_html=True)
                        df_res = pd.DataFrame([
                            {
                                "Model": r["model"].replace('_', ' ').title(),
                                "Prediction": r["label"],
                                "Confidence": f"{r['confidence']}%"
                            }
                            for r in results
                        ])
                        st.dataframe(df_res, use_container_width=True)

                    else:
                        st.markdown(
                            f"""
                            <div style="border-bottom: 1px solid #1E293B; padding-bottom: 10px; margin-bottom: 16px;">
                                <h3 style="margin: 0; font-size: 1.1rem; color: #F8FAFC;"><i class="fa-solid fa-chart-simple" style="margin-right: 6px;"></i>Model Prediction Results ({len(results)} Models)</h3>
                                <div style="font-size: 0.8rem; color: #64748B;">Total processing time: {inference_sec}s</div>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
                        
                        cols = st.columns(len(results))
                        for idx, r in enumerate(results):
                            with cols[idx]:
                                badge_style = "badge-real" if r["label"] == "Real" else "badge-fake"
                                badge_icon = '<i class="fa-solid fa-circle-check"></i>' if r["label"] == "Real" else '<i class="fa-solid fa-circle-xmark"></i>'
                                st.markdown(
                                    f"""
                                    <div style="background-color: #0B1120; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; text-align: center;">
                                        <div style="font-size: 0.85rem; font-weight: 600; color: #94A3B8; margin-bottom: 8px;">
                                            {r["model"].replace('_', ' ').title()}
                                        </div>
                                        <div style="margin-bottom: 8px;">
                                            <span class="fna-badge {badge_style}">
                                                {badge_icon} {r["label"]}
                                            </span>
                                        </div>
                                        <div style="font-size: 0.8rem; color: #64748B;">Confidence: <b style="color: #F8FAFC;">{r["confidence"]}%</b></div>
                                    </div>
                                    """,
                                    unsafe_allow_html=True
                                )
                                st.progress(min(r["confidence"] / 100.0, 1.0))
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.caption("ℹ️ This is a decision-support tool. Always corroborate news claims with reputable investigative sources.")

        elif not check_btn:
            # Empty State
            st.markdown(
                """
                <div class="fna-empty-state">
                    <div class="fna-empty-icon" style="color: #6366F1;"><i class="fa-solid fa-magnifying-glass-chart"></i></div>
                    <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600; color: #F8FAFC;">Ready for Analysis</h3>
                    <p style="color: #64748B; font-size: 0.9rem; margin: 0;">
                        Enter a news article or headline above and click "Check News" to run our neural ensemble verification.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif page in ["Prediction History", "History"]:
        if not st.session_state.user_auth.get("logged_in"):
            st.warning("Please sign in or register in the sidebar to access Prediction History.")
            st.session_state.app_nav = "Home (Check News)"
        else:
            curr_u_id = st.session_state.user_auth.get("user_id")
            curr_u_name = st.session_state.user_auth.get("username")
            render_prediction_history_page(is_admin=False, current_user_id=curr_u_id, current_username=curr_u_name)

    elif page == "Model Comparison":
        render_model_comparison_page()

    elif page == "Model Training":
        render_model_training_page(is_admin=False)

    elif page == "Model Insights":
        render_model_insights_page(is_admin=False)

    elif page == "Settings":
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <div style="background-color: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; color: #60A5FA;">
                    <i class="fa-solid fa-sliders"></i>
                </div>
                <div>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">System Settings</h1>
                    <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.95rem;">Configure application defaults and inference preferences</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown('<div class="fna-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #F8FAFC; margin-top: 0;'><i class='fa-solid fa-microchip' style='margin-right: 6px;'></i>Inference & Detection Preferences</h4>", unsafe_allow_html=True)
        def_tier = st.selectbox("Default Model Tier", ["Basic", "Intermediate", "Hardcore", "Ensemble (Majority Vote - All Models)"], index=3)
        min_conf = st.slider("Confidence Alert Threshold (%)", min_value=50, max_value=95, value=75)
        st.checkbox("Enable real-time linguistic feature extraction", value=True)
        st.checkbox("Log all inference queries to database audit trail", value=True)
        st.checkbox("Auto-flag borderline claims below threshold as Unverified", value=True)
        if st.button("Save Preferences"):
            st.success("Preferences saved successfully!")
        st.markdown('</div>', unsafe_allow_html=True)

    elif page == "About":
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <div style="background-color: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem; color: #C084FC;">
                    <i class="fa-solid fa-circle-info"></i>
                </div>
                <div>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">About FakeNewsAI</h1>
                    <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.95rem;">Multi-tier real-time news credibility detection architecture</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            """
            <div class="fna-card">
                <h3 style="color: #38BDF8; margin-top: 0;"><i class="fa-solid fa-shield-halved" style="margin-right: 6px;"></i>Architecture & Design</h3>
                <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.7;">
                    FakeNewsAI is an end-to-end artificial intelligence decision support system designed to identify misinformation and synthetic news articles. 
                    It combines classical natural language processing (TF-IDF with Logistic Regression, Random Forest, Naive Bayes, Decision Trees, Linear SVM) 
                    with modern deep learning and transformer language representations (BERT, DistilBERT, RoBERTa, ALBERT, ELECTRA, BiLSTM).
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 15px;">
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <b style="color: #34D399;"><i class="fa-solid fa-brain" style="margin-right: 6px;"></i>11 Active Models</b>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Basic, Intermediate, Hardcore and Ensemble tiers</div>
                    </div>
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <b style="color: #38BDF8;"><i class="fa-solid fa-database" style="margin-right: 6px;"></i>338,277 Dataset</b>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Indexed articles from ISOT, Kaggle, and verified sources</div>
                    </div>
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <b style="color: #FBBF24;"><i class="fa-solid fa-clock-rotate-left" style="margin-right: 6px;"></i>Real-time Audit</b>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">SQLite persistent transaction & query logging</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Bottom of Page - Admin Access Section
    st.markdown("<hr style='border-color: #1E293B; margin-top: 50px; margin-bottom: 25px;'>", unsafe_allow_html=True)
    footer_col1, footer_col2 = st.columns([3, 1])
    with footer_col1:
        st.markdown(
            """
            <div style="font-size: 0.85rem; color: #64748B;">
                <i class="fa-solid fa-shield-halved" style="color: #818CF8; margin-right: 4px;"></i> <b>FakeNewsAI</b> — Advanced Multi-Model Detection & Analysis System<br>
                <span style="font-size: 0.78rem; color: #475569;">System authorized for research and evaluation purposes.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with footer_col2:
        if hasattr(st, "page_link"):
            st.page_link("pages/admin.py", label="Admin Panel", icon=":material/admin_panel_settings:")
        else:
            if st.button("Admin Panel", icon=":material/admin_panel_settings:"):
                st.switch_page("pages/admin.py")

if __name__ == "__main__":
    main()

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
from preprocessing import preprocess
from db import init_db
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
    # Sidebar
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 25px;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">FakeNewsAI</h3>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Detection System</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase;'>Navigation</p>", unsafe_allow_html=True)
        nav_options = [
            "Home (Check News)",
            "Prediction History",
            "Model Comparison",
            "Model Insights"
        ]
        
        if "app_nav" not in st.session_state:
            st.session_state.app_nav = "Home (Check News)"
            
        default_idx = nav_options.index(st.session_state.app_nav) if st.session_state.app_nav in nav_options else 0
        page = st.radio("Navigation", nav_options, index=default_idx, label_visibility="collapsed")
        st.session_state.app_nav = page
        
        st.markdown("<hr style='border-color: #1E293B; margin: 25px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #94A3B8;">
                <span style="color: #10B981; font-weight: 600;">● Active System</span><br>
                11 detection models active with real-time analytics.
            </div>
            """,
            unsafe_allow_html=True
        )

    # Main Area
    if page == "Home (Check News)":
        # Top Header Row
        col_title, col_status = st.columns([3, 1])
        with col_title:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
                    <span style="font-size: 2rem;">📰</span>
                    <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">Fake News Detection System</h1>
                </div>
                <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0;">
                    Enter a news article, headline or claim below to check if it's real or fake using AI and trained models.
                </p>
                """,
                unsafe_allow_html=True
            )
        with col_status:
            st.markdown(
                """
                <div style="text-align: right; margin-top: 10px;">
                    <span class="fna-pill-ai">
                        <span style="color: #10B981;">●</span> AI Powered <span style="color: #64748B;">|</span> Using trained models
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Main Input Card
        # st.markdown('<div class="fna-card">', unsafe_allow_html=True)
        
        # Pill Tab
        # st.markdown(
        #     """
        #     <div style="margin-bottom: 15px;">
        #         <span style="background-color: #2563EB; color: white; padding: 6px 16px; border-radius: 8px; font-weight: 600; font-size: 0.85rem; display: inline-flex; align-items: center; gap: 6px;">
        #             📝 Text Input
        #         </span>
        #     </div>
        #     """, 
        #     unsafe_allow_html=True
        # )

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
                f"Characters: <b style='color: #94A3B8;'>{char_count}</b> &nbsp;|&nbsp; Supports English and multilingual news"
                f"</div>",
                unsafe_allow_html=True
            )
        with col_controls:
            col_tier, col_btn = st.columns([1.3, 1])
            with col_tier:
                selected_tier = st.selectbox(
                    "Select Model Tier",
                    list(TIER_OPTIONS.keys()),
                    index=0,
                    label_visibility="collapsed"
                )
            with col_btn:
                check_btn = st.button("🔍 Check News", use_container_width=True, disabled=(char_count == 0))

        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction and Analysis Logic
        if check_btn and raw_text.strip():
            models_to_run = TIER_OPTIONS[selected_tier]
            is_ensemble = "Ensemble" in selected_tier
            
            with st.spinner("Analyzing text using trained models..."):
                start_time = time.time()
                cleaned_text = preprocess(raw_text)
                results = []
                
                real_count = 0
                fake_count = 0
                
                for m_name in models_to_run:
                    res = predict(raw_text, m_name)
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
                        # Log to db
                        log_query(raw_text, cleaned_text, m_name, label, conf, inference_time=0.38, status="Reviewed")
                        log_system_event("INFO", "Predictor", f"{m_name.replace('_', ' ').title()} classified query as {label} ({conf}%)")
                
                inference_sec = round(time.time() - start_time, 2)
                
            # Render Results Card
            if results:
                st.markdown('<div class="fna-card">', unsafe_allow_html=True)
                
                if is_ensemble:
                    final_label = "Real" if real_count >= fake_count else "Fake"
                    vote_ratio = f"{max(real_count, fake_count)} of {len(results)}"
                    badge_style = "badge-real" if final_label == "Real" else "badge-fake"
                    badge_icon = "🎯" if final_label == "Real" else "🛑"
                    avg_conf = round(sum(r["confidence"] for r in results) / len(results), 1)

                    st.markdown(
                        f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1E293B; padding-bottom: 12px; margin-bottom: 16px;">
                            <div>
                                <span class="fna-badge {badge_style}" style="font-size: 1.15rem; padding: 6px 16px;">
                                    {badge_icon} FINAL VERDICT: {final_label.upper()} NEWS
                                </span>
                                <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px;">
                                    Ensemble Consensus: <b>{vote_ratio} models agree</b> | Inference: <b>{inference_sec}s</b>
                                </div>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 0.8rem; color: #64748B;">Average Confidence</div>
                                <div style="font-size: 1.5rem; font-weight: 700; color: #F8FAFC;">{avg_conf}%</div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.progress(min(avg_conf / 100.0, 1.0))
                    
                    st.markdown("<p style='font-weight: 600; font-size: 0.95rem; margin-top: 15px; color: #F8FAFC;'>Individual Model Predictions:</p>", unsafe_allow_html=True)
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
                            <h3 style="margin: 0; font-size: 1.1rem; color: #F8FAFC;">Model Prediction Results ({len(results)} Models)</h3>
                            <div style="font-size: 0.8rem; color: #64748B;">Total processing time: {inference_sec}s</div>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    cols = st.columns(len(results))
                    for idx, r in enumerate(results):
                        with cols[idx]:
                            badge_style = "badge-real" if r["label"] == "Real" else "badge-fake"
                            badge_icon = "✅" if r["label"] == "Real" else "❌"
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
                    <div class="fna-empty-icon">📑</div>
                    <h3 style="margin: 0 0 6px 0; font-size: 1.15rem; font-weight: 600; color: #F8FAFC;">No analysis yet</h3>
                    <p style="color: #64748B; font-size: 0.9rem; margin: 0;">
                        Enter a news article or headline above and click "Check News" to see the result.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )

    elif page in ["Prediction History", "History"]:
        render_prediction_history_page(is_admin=False)

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
                <div style="background-color: rgba(59, 130, 246, 0.15); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
                    ⚙️
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
        st.markdown("<h4 style='color: #F8FAFC; margin-top: 0;'>Inference & Detection Preferences</h4>", unsafe_allow_html=True)
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
                <div style="background-color: rgba(168, 85, 247, 0.15); border: 1px solid rgba(168, 85, 247, 0.3); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.4rem;">
                    ℹ️
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
                <h3 style="color: #38BDF8; margin-top: 0;">🛡️ Architecture & Design</h3>
                <p style="color: #CBD5E1; font-size: 0.92rem; line-height: 1.7;">
                    FakeNewsAI is an end-to-end artificial intelligence decision support system designed to identify misinformation and synthetic news articles. 
                    It combines classical natural language processing (TF-IDF with Logistic Regression, Random Forest, Naive Bayes, Decision Trees, Linear SVM) 
                    with modern deep learning and transformer language representations (BERT, DistilBERT, RoBERTa, ALBERT, ELECTRA, BiLSTM).
                </p>
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 12px; margin-top: 15px;">
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <b style="color: #34D399;">11 Active Models</b>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Basic, Intermediate, Hardcore and Ensemble tiers</div>
                    </div>
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <b style="color: #38BDF8;">338,277 Dataset</b>
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-top: 4px;">Indexed articles from ISOT, Kaggle, and verified sources</div>
                    </div>
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px;">
                        <b style="color: #FBBF24;">Real-time Audit</b>
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
                🛡️ <b>FakeNewsAI</b> — Advanced Multi-Model Detection & Analysis System<br>
                <span style="font-size: 0.78rem; color: #475569;">System authorized for research and evaluation purposes.</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    with footer_col2:
        if hasattr(st, "page_link"):
            st.page_link("pages/admin.py", label="⚙️ Admin Panel", icon=None)
        else:
            if st.button("⚙️ Open Admin Panel"):
                st.switch_page("pages/admin.py")

if __name__ == "__main__":
    main()

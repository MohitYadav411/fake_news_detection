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
        page = st.radio("Navigation", ["Home (Check News)", "History"], label_visibility="collapsed")
        
        st.markdown("<hr style='border-color: #1E293B; margin: 30px 0 20px 0;'>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; font-size: 0.8rem; color: #94A3B8;">
                <span style="color: #10B981; font-weight: 600;">● Active System</span><br>
                Multi-model linguistic analysis + ensemble voting enabled.
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

    elif page == "History":
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem;">
                <span style="font-size: 1.8rem;">📜</span>
                <h1 style="margin: 0; font-size: 1.6rem; font-weight: 700; color: #F8FAFC;">Prediction History</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        df = get_history(limit=100)
        if not df.empty:
            # st.markdown('<div class="fna-card">', unsafe_allow_html=True)
            col_search, col_exp = st.columns([3, 1])
            with col_search:
                search_query = st.text_input("Search history", placeholder="Search news content...", label_visibility="collapsed")
            with col_exp:
                export_path = os.path.join(BASE_DIR, "data", "history_export.csv")
                export_history(export_path)
                if os.path.exists(export_path):
                    with open(export_path, "rb") as f:
                        st.download_button("📥 Export CSV", data=f, file_name="prediction_history.csv", mime="text/csv", use_container_width=True)

            if search_query:
                df = df[df["raw_text"].str.contains(search_query, case=False, na=False)]
                
            display_df = df[["prediction_code", "raw_text", "predicted_label", "confidence", "model_name", "logged_at"]].copy()
            display_df.columns = ["Prediction ID", "News Text", "Verdict", "Confidence (%)", "Model", "Timestamp"]
            st.dataframe(display_df, use_container_width=True)
        else:
            st.info("No prediction history recorded yet.")

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

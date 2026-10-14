"""
pages/admin.py
FakeNewsAI Admin Dashboard & Review Queue.
Implements the high-fidelity dark analytics interface matching Screenshots 2 & 3.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import sys
import subprocess
import matplotlib.pyplot as plt

# Ensure src is on sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, "src"))

from db import init_db, get_connection
from history import get_kpis, get_history, update_review_status, export_history, get_dataset_stats
from ui_theme import inject_theme, render_kpi, render_badge

init_db()

# Page config
st.set_page_config(
    page_title="FakeNewsAI - Admin Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply global dark theme styling
inject_theme()

# Model list for evaluation
EVAL_MODELS = [
    "random_forest", "naive_bayes", "logistic_regression", "svm",
    "decision_tree", "bert", "distilbert", "roberta", "albert", "electra", "lstm"
]

def load_metrics(model_name):
    metrics_path = os.path.join(BASE_DIR, "models", f"{model_name}_metrics.json")
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def render_admin_header():
    col_head, col_user = st.columns([3, 1])
    with col_head:
        st.markdown(
            """
            <div style="margin-bottom: 20px;">
                <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">Admin Dashboard</h1>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                    Monitor, analyze and manage your Fake News Detection System
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_user:
        st.markdown(
            """
            <div style="display: flex; align-items: center; justify-content: flex-end; margin-top: 8px;">
                <span class="fna-pill-ai" style="padding: 5px 12px; font-size: 0.8rem;">
                    <span style="color: #10B981;">●</span> System Online
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    # Admin Authentication
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False

    if not st.session_state.admin_authenticated:
        _, login_center_col, _ = st.columns([1, 1.2, 1])
        with login_center_col:
            st.markdown(
                """
                <div style="background-color: #0F172A; border: 1px solid #1E293B; border-radius: 12px; padding: 2rem; margin-top: 60px; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35); text-align: center;">
                    <span style="font-size: 2.4rem;">🛡️</span>
                    <h2 style="margin: 12px 0 6px 0; color: #F8FAFC; font-size: 1.4rem; font-weight: 700;">Admin Portal</h2>
                    <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 20px;">Please enter password to access the panel.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            pwd = st.text_input("Password", type="password", placeholder="Enter admin password", label_visibility="collapsed")
            if st.button("Unlock Dashboard", use_container_width=True):
                if pwd == "admin123":
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        st.stop()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 25px;">
                <span style="font-size: 2rem;">🛡️</span>
                <div>
                    <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700;">FakeNewsAI</h3>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Admin Panel</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("<p style='font-size: 0.8rem; font-weight: 600; color: #64748B; text-transform: uppercase;'>Admin Navigation</p>", unsafe_allow_html=True)
        admin_page = st.radio(
            "Admin Section",
            ["Overview (Dashboard)", "Prediction History"],
            label_visibility="collapsed"
        )
        
        st.markdown("<hr style='border-color: #1E293B; margin: 30px 0 20px 0;'>", unsafe_allow_html=True)
        if hasattr(st, "page_link"):
            st.page_link("app.py", label="🏠 Back to Main App", icon=None)
        else:
            if st.button("🏠 Back to Main App", use_container_width=True):
                st.switch_page("app.py")
                
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()

    # Fetch KPIs & Data (100% Real-Time from Database)
    kpis = get_kpis()
    display_total = kpis["total"]
    display_real = kpis["real"]
    display_fake = kpis["fake"]
    display_unverified = kpis["unverified"]
    
    real_pct = round((display_real / display_total) * 100, 1) if display_total > 0 else 0.0
    fake_pct = round((display_fake / display_total) * 100, 1) if display_total > 0 else 0.0
    unv_pct = round((display_unverified / display_total) * 100, 1) if display_total > 0 else 0.0

    # PAGE 1: Overview
    if admin_page == "Overview (Dashboard)":
        render_admin_header()

        # Top 6 KPI Cards (Live Real-Time Values)
        kpi_cols = st.columns(6)
        with kpi_cols[0]:
            render_kpi("Total Predictions", f"{display_total:,}", "Live query records", "#3B82F6", "📄")
        with kpi_cols[1]:
            render_kpi("Real News", f"{display_real:,}", f"{real_pct}% of total", "#10B981", "✅")
        with kpi_cols[2]:
            render_kpi("Fake News", f"{display_fake:,}", f"{fake_pct}% of total", "#EF4444", "❌")
        with kpi_cols[3]:
            render_kpi("Unverified", f"{display_unverified:,}", f"{unv_pct}% of total", "#F59E0B", "❓")
        with kpi_cols[4]:
            render_kpi("Avg. Inference Time", f"{kpis['avg_inference']}s", "Real-time latency", "#A855F7", "⚡")
        with kpi_cols[5]:
            render_kpi("Best Model (F1)", f"{kpis['best_f1']}", f"{kpis['best_model']} 🏆", "#3B82F6", "⚙️")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Middle Row: Model Performance & Confusion Matrix
        col_perf, col_matrix = st.columns([1.3, 1])
        
        with col_perf:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">📊 Model Performance Comparison</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            
            # Identify model with highest F1 dynamically
            best_f1_key = None
            max_f1 = -1.0
            for m in EVAL_MODELS:
                mets = load_metrics(m)
                if mets and mets.get("f1", 0) > max_f1:
                    max_f1 = mets.get("f1", 0)
                    best_f1_key = m

            perf_list = []
            for idx, m in enumerate(EVAL_MODELS):
                mets = load_metrics(m)
                if mets:
                    perf_list.append({
                        "#": idx + 1,
                        "Model": m.replace('_', ' ').title(),
                        "Accuracy": round(mets.get("accuracy", 0), 3),
                        "Precision": round(mets.get("precision", 0), 3),
                        "Recall": round(mets.get("recall", 0), 3),
                        "F1-Score": round(mets.get("f1", 0), 3),
                        "Status": "🏆 Best" if m == best_f1_key else "Active"
                    })
            if perf_list:
                df_perf = pd.DataFrame(perf_list)
                st.dataframe(df_perf, use_container_width=True, hide_index=True)
            else:
                st.info("No metrics found. Train models to display benchmark comparisons.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_matrix:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">📈 Confusion Matrix & Report</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            selected_m = st.selectbox(
                "Model",
                [m.replace('_', ' ').title() for m in EVAL_MODELS],
                index=0,
                label_visibility="collapsed"
            )
            raw_model_key = selected_m.lower().replace(' ', '_')
            img_path = os.path.join(BASE_DIR, "models", f"{raw_model_key}_confusion_matrix.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{selected_m} Confusion Matrix", use_container_width=True)
            else:
                st.info(f"Confusion matrix image for {selected_m} not found.")

            # Real Model Evaluation Report
            st.markdown(f"<div style='font-size: 0.85rem; font-weight: 600; color: #94A3B8; margin-top: 10px;'>Evaluated Metrics ({selected_m})</div>", unsafe_allow_html=True)
            sel_mets = load_metrics(raw_model_key) or {}
            acc_val = round(sel_mets.get("accuracy", 0.0), 3)
            prec_val = round(sel_mets.get("precision", 0.0), 3)
            rec_val = round(sel_mets.get("recall", 0.0), 3)
            f1_val = round(sel_mets.get("f1", 0.0), 3)

            report_df = pd.DataFrame([
                {"Metric": "Accuracy", "Score": acc_val, "Status": "Evaluated"},
                {"Metric": "Precision", "Score": prec_val, "Status": "Evaluated"},
                {"Metric": "Recall", "Score": rec_val, "Status": "Evaluated"},
                {"Metric": "F1-Score", "Score": f1_val, "Status": "Evaluated"}
            ])
            st.dataframe(report_df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Third Row: Word Clouds & Prediction Distribution
        col_wc, col_dist = st.columns([1.3, 1])
        
        with col_wc:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">☁️ Word Clouds (Text Analysis)</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            wc_type = st.radio("Type", ["Real News", "Fake News"], horizontal=True, label_visibility="collapsed")
            wc_file = "wordcloud_true.png" if wc_type == "Real News" else "wordcloud_fake.png"
            wc_path = os.path.join(BASE_DIR, "models", wc_file)
            if os.path.exists(wc_path):
                st.image(wc_path, caption=f"Frequent Terms - {wc_type}", use_container_width=True)
            else:
                st.info(f"{wc_type} word cloud image not found in models directory.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_dist:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">🍩 Prediction Distribution</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            # Generate Matplotlib Donut Chart matching dark UI
            fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0F172A")
            ax.set_facecolor("#0F172A")
            
            if display_total > 0:
                sizes = [display_real, display_fake, display_unverified]
                colors = ["#10B981", "#EF4444", "#F59E0B"]
                labels = [f"Real: {real_pct}%", f"Fake: {fake_pct}%", f"Unverified: {unv_pct}%"]
                
                pie_sizes = []
                pie_colors = []
                pie_labels = []
                for s, c, l in zip(sizes, colors, labels):
                    if s > 0:
                        pie_sizes.append(s)
                        pie_colors.append(c)
                        pie_labels.append(l)

                wedges, _ = ax.pie(
                    pie_sizes, 
                    colors=pie_colors, 
                    startangle=90, 
                    wedgeprops=dict(width=0.38, edgecolor="#0F172A", linewidth=2)
                )
                
                ax.text(0, 0.05, f"{display_total:,}", ha='center', va='center', fontsize=15, fontweight='bold', color='#F8FAFC')
                ax.text(0, -0.18, "Total", ha='center', va='center', fontsize=10, color='#94A3B8')
                ax.legend(wedges, pie_labels, loc="lower center", bbox_to_anchor=(0.5, -0.2), frameon=False, labelcolor="#CBD5E1", fontsize=8.5)
            else:
                ax.text(0, 0, "No Predictions Yet", ha='center', va='center', fontsize=12, color='#94A3B8')
                ax.axis('off')
            
            st.pyplot(fig)
            plt.close(fig)
            st.markdown('</div>', unsafe_allow_html=True)

        # Bottom Row: Review Queue & Dataset / Retraining
        col_queue, col_data_ops = st.columns([1.3, 1])
        
        with col_queue:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">📋 Recent Predictions / Review Queue</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            
            hist_df = get_history(limit=8)
            if not hist_df.empty:
                q_rows = []
                for i, row in hist_df.iterrows():
                    short_text = str(row['raw_text'])[:45] + "..." if len(str(row['raw_text'])) > 45 else str(row['raw_text'])
                    q_rows.append({
                        "#": i + 1,
                        "News Text (short)": short_text,
                        "Prediction": row['predicted_label'],
                        "Confidence": f"{round(float(row['confidence']), 2)}%",
                        "Model": str(row['model_name']).replace('_', ' ').title(),
                        "Date": str(row['logged_at'])[:10],
                        "Status": row.get('status', 'Reviewed')
                    })
                st.dataframe(pd.DataFrame(q_rows), use_container_width=True, hide_index=True)
            else:
                st.info("No queries logged yet.")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_data_ops:
            # Real-Time Dataset Summary Card
            dstats = get_dataset_stats()
            st.markdown(
                f"""
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">💾 Dataset Summary</div>
                    </div>
                    <div style="font-size: 0.85rem; line-height: 1.8;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Total Records:</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{dstats['total_records']:,}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Fake Articles:</span>
                            <span style="color: #F87171; font-weight: 600;">{dstats['fake_records']:,} ({dstats['fake_pct']}%)</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Real Articles:</span>
                            <span style="color: #34D399; font-weight: 600;">{dstats['true_records']:,} ({dstats['true_pct']}%)</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Duplicate Rows:</span>
                            <span style="color: #F8FAFC;">{dstats['duplicate_rows']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Last Updated:</span>
                            <span style="color: #F8FAFC;">{dstats['last_updated']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Version:</span>
                            <span style="color: #38BDF8; font-weight: 600;">{dstats['version']} (Active)</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Model Retraining Card (Live Model Reference)
            st.markdown(
                f"""
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">🔄 Model Retraining</div>
                    </div>
                    <div style="font-size: 0.85rem; line-height: 1.8; margin-bottom: 15px;">
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Top Performer:</span>
                            <span style="color: #F8FAFC; font-weight: 600;">{kpis['best_model']}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #94A3B8;">Training Status:</span>
                            <span style="color: #34D399; font-weight: 600;">Ready</span>
                        </div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            
            if st.button("🚀 Start Retraining", use_container_width=True):
                with st.spinner("Retraining basic models via train.py..."):
                    proc = subprocess.run(
                        [sys.executable, os.path.join(BASE_DIR, "src", "train.py"), "--tier", "all"],
                        cwd=BASE_DIR, capture_output=True, text=True
                    )
                    if proc.returncode == 0:
                        st.success("Retraining completed successfully!")
                    else:
                        st.error("Retraining failed. Check dependencies or data files.")
                        st.code(proc.stderr or proc.stdout)
            st.markdown('</div>', unsafe_allow_html=True)

    # PAGE 2: Prediction History & Details Inspector (Screenshot 3)
    elif admin_page == "Prediction History":
        render_admin_header()

        # Top 4 KPI Cards
        top4 = st.columns(4)
        with top4[0]:
            render_kpi("Total Predictions", f"{display_total:,}", "Live query records", "#3B82F6", "📄")
        with top4[1]:
            render_kpi("Real News", f"{display_real:,}", f"{real_pct}% of total", "#10B981", "✅")
        with top4[2]:
            render_kpi("Fake News", f"{display_fake:,}", f"{fake_pct}% of total", "#EF4444", "❌")
        with top4[3]:
            render_kpi("Unverified", f"{display_unverified:,}", f"{unv_pct}% of total", "#F59E0B", "❓")

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        col_hist_table, col_hist_detail = st.columns([1.6, 1])

        hist_df = get_history(limit=50)

        with col_hist_table:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">📑 Prediction History</div>
                    </div>
                """,
                unsafe_allow_html=True
            )

            # Modern Filter Bar
            col_search, col_f_model, col_f_status = st.columns([2, 1.2, 1.2])
            with col_search:
                search_term = st.text_input("Search", placeholder="🔍 Search news content or ID...", label_visibility="collapsed")
            with col_f_model:
                f_model = st.selectbox("Model Filter", ["All Models"] + [m.replace('_', ' ').title() for m in EVAL_MODELS], label_visibility="collapsed")
            with col_f_status:
                f_status = st.selectbox("Status Filter", ["All Statuses", "Verified", "Reviewed", "Pending"], label_visibility="collapsed")

            # Apply Filters
            filtered_df = hist_df.copy()
            if not filtered_df.empty:
                if search_term:
                    filtered_df = filtered_df[
                        filtered_df['raw_text'].str.contains(search_term, case=False, na=False) |
                        filtered_df['prediction_code'].str.contains(search_term, case=False, na=False)
                    ]
                if f_model != "All Models":
                    filtered_df = filtered_df[filtered_df['model_name'].str.lower() == f_model.lower().replace(' ', '_')]
                if f_status != "All Statuses":
                    filtered_df = filtered_df[filtered_df['status'].str.lower() == f_status.lower()]

            selected_row = None
            if not filtered_df.empty:
                # Add a dropdown or native table selection for details
                col_sel_lbl, col_sel_box = st.columns([1, 2.5])
                with col_sel_lbl:
                    st.markdown("<p style='font-size: 0.85rem; color: #94A3B8; margin-top: 6px; font-weight: 500;'>Select to inspect:</p>", unsafe_allow_html=True)
                with col_sel_box:
                    options_map = {
                        f"{row['prediction_code']} | {str(row['raw_text'])[:38]}...": row['prediction_code']
                        for _, row in filtered_df.iterrows()
                    }
                    selected_key = st.selectbox(
                        "Inspect record",
                        options=list(options_map.keys()),
                        index=0,
                        label_visibility="collapsed"
                    )
                    selected_code = options_map[selected_key]
                    selected_row = filtered_df[filtered_df['prediction_code'] == selected_code].iloc[0]

                # Clean structured table display
                table_display = filtered_df[["prediction_code", "raw_text", "predicted_label", "confidence", "model_name", "status"]].copy()
                table_display.columns = ["Prediction ID", "News Text (Preview)", "Verdict", "Confidence (%)", "Model", "Status"]
                st.dataframe(
                    table_display,
                    use_container_width=True,
                    hide_index=True,
                    height=380
                )
                st.caption(f"Showing {len(filtered_df)} records matching filter.")
            else:
                st.info("No records matched the filter criteria.")

            st.markdown('</div>', unsafe_allow_html=True)

        with col_hist_detail:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title">👁️ Prediction Details</div>
                    </div>
                """,
                unsafe_allow_html=True
            )

            if selected_row is not None:
                p_code = selected_row["prediction_code"]
                p_label = str(selected_row["predicted_label"])
                p_conf = round(float(selected_row["confidence"]), 2)
                p_model = str(selected_row["model_name"]).replace('_', ' ').title()
                p_text = selected_row["raw_text"]
                p_time = selected_row["logged_at"]
                p_res_id = selected_row["result_id"]
                p_status = selected_row.get("status", "Reviewed")

                st.markdown(
                    f"""
                    <div style="font-size: 0.85rem; margin-bottom: 12px;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                            <span style="color: #94A3B8;">Prediction ID:</span>
                            <span style="color: #38BDF8; font-weight: 700; font-family: monospace;">{p_code}</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 12px;">
                            <span style="color: #94A3B8;">Date & Time:</span>
                            <span style="color: #CBD5E1;">{p_time}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div style='font-size: 0.8rem; color: #94A3B8; margin-bottom: 4px; font-weight: 500;'>Full News Text:</div>", unsafe_allow_html=True)
                st.markdown(
                    f"""
                    <div style="background-color: #070D18; border: 1px solid #1E293B; border-radius: 8px; padding: 12px; font-size: 0.85rem; color: #E2E8F0; max-height: 150px; overflow-y: auto; margin-bottom: 16px; line-height: 1.45;">
                        {p_text}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                badge_class = "badge-real" if p_label.lower() == "real" else "badge-fake"
                icon = "✅" if p_label.lower() == "real" else "🛑"
                verdict_sub = "This news is likely to be real and credible." if p_label.lower() == "real" else "This news contains suspicious or fabricated claims."

                st.markdown(
                    f"""
                    <div style="margin-bottom: 14px;">
                        <div style="font-size: 0.8rem; color: #94A3B8; margin-bottom: 6px; font-weight: 500;">Prediction Result:</div>
                        <div class="fna-badge {badge_class}" style="font-size: 0.95rem; padding: 6px 14px; margin-bottom: 6px;">
                            {icon} {p_label.title()} News
                        </div>
                        <div style="font-size: 0.8rem; color: #64748B;">{verdict_sub}</div>
                    </div>
                    <div style="margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 4px;">
                            <span style="color: #94A3B8;">Confidence Score:</span>
                            <span style="font-weight: 700; color: #F8FAFC;">{p_conf}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.progress(min(p_conf / 100.0, 1.0))

                st.markdown(
                    f"""
                    <div style="margin: 14px 0 20px 0; font-size: 0.85rem; display: flex; justify-content: space-between;">
                        <span style="color: #94A3B8;">Model Used:</span>
                        <span style="font-weight: 600; color: #F8FAFC;">{p_model}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if st.button("✅ Mark as Reviewed", use_container_width=True):
                    if update_review_status(p_res_id, "Reviewed"):
                        st.success(f"{p_code} marked as Reviewed!")
                        st.rerun()

            else:
                st.info("Select a prediction from the table on the left to inspect its details.")

            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

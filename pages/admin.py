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

from db import init_db, get_connection, verify_password, get_admin_password_hash, update_admin_password
from generate_wordclouds import generate_wordclouds
from history import get_kpis, get_history, update_review_status, export_history, get_dataset_stats
from ui_theme import inject_theme, render_kpi, render_badge
from views import (
    render_model_insights_page,
    render_prediction_history_page,
    render_model_comparison_page,
    render_dataset_manager_page,
    render_model_training_page,
    render_system_logs_page
)

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
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; margin-bottom: 20px;">
            <div style="min-width: 250px; flex: 1 1 280px;">
                <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">Admin Dashboard</h1>
                <p style="margin: 4px 0 0 0; color: #94A3B8; font-size: 0.9rem;">
                    Monitor, analyze and manage your Fake News Detection System
                </p>
            </div>
            <div>
                <span class="fna-pill-ai" style="padding: 5px 12px; font-size: 0.8rem;">
                    <span style="color: #10B981;">●</span> System Online
                </span>
            </div>
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
                    <div style="display: inline-flex; align-items: center; justify-content: center; width: 64px; height: 64px; border-radius: 16px; background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(6, 182, 212, 0.15) 100%); border: 1px solid rgba(99, 102, 241, 0.4); font-size: 2rem; color: #818CF8; box-shadow: 0 0 20px rgba(99, 102, 241, 0.3); margin-bottom: 8px;">
                        <i class="fa-solid fa-lock"></i>
                    </div>
                    <h2 style="margin: 12px 0 6px 0; color: #F8FAFC; font-size: 1.4rem; font-weight: 700;">Admin Portal</h2>
                    <p style="color: #64748B; font-size: 0.85rem; margin-bottom: 20px;">Please enter password to access the panel.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            pwd = st.text_input("Password", type="password", placeholder="Enter admin password", label_visibility="collapsed")
            if st.button("Unlock Dashboard", use_container_width=True):
                admin_hash = get_admin_password_hash()
                if verify_password(pwd, admin_hash):
                    st.session_state.admin_authenticated = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
        st.stop()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown(
            """
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 28px; padding-bottom: 15px; border-bottom: 1px solid rgba(51, 65, 85, 0.6);">
                <div style="background: linear-gradient(135deg, rgba(99, 102, 241, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%); border: 1px solid rgba(99, 102, 241, 0.5); border-radius: 12px; width: 44px; height: 44px; display: flex; align-items: center; justify-content: center; font-size: 1.35rem; color: #818CF8; box-shadow: 0 0 16px rgba(99, 102, 241, 0.35);">
                    <i class="fa-solid fa-shield-halved"></i>
                </div>
                <div>
                    <h3 style="margin: 0; font-size: 1.3rem; font-weight: 800; background: linear-gradient(135deg, #FFFFFF 0%, #CBD5E1 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">FakeNewsAI</h3>
                    <div style="font-size: 0.78rem; color: #818CF8; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase;">Admin Panel</div>
                </div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        
        st.markdown("<p style='font-size: 0.78rem; font-weight: 700; color: #6D4AFF; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 12px;'><i class='fa-solid fa-terminal' style='margin-right: 6px;'></i>Panel Menu</p>", unsafe_allow_html=True)
        
        admin_nav_items = [
            ("Overview (Dashboard)", "Overview", ":material/dashboard:"),
            ("User Accounts", "User Database", ":material/group:"),
            ("Prediction History", "Prediction History", ":material/history:"),
            ("Settings", "Change Password", ":material/settings:")
        ]
        
        if "admin_nav_sel" not in st.session_state:
            st.session_state.admin_nav_sel = "Overview (Dashboard)"
            
        for page_key, display_label, icon_name in admin_nav_items:
            is_active = (st.session_state.admin_nav_sel == page_key)
            btn_type = "primary" if is_active else "secondary"
            if st.button(display_label, key=f"admin_nav_{page_key}", type=btn_type, icon=icon_name, use_container_width=True):
                st.session_state.admin_nav_sel = page_key
                st.rerun()

        admin_page = st.session_state.admin_nav_sel
        
        st.markdown("<hr style='border-color: #1E293B; margin: 30px 0 20px 0;'>", unsafe_allow_html=True)
        if hasattr(st, "page_link"):
            st.page_link("app.py", label="Back to Main App", icon=":material/arrow_back:")
        else:
            if st.button("Back to Main Page", icon=":material/arrow_back:", use_container_width=True):
                st.switch_page("app.py")
                
        if st.button("Logout", icon=":material/logout:", use_container_width=True):
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

        # Top 5 KPI Cards (Live Real-Time Values)
        kpi_cols = st.columns(5)
        with kpi_cols[0]:
            render_kpi("Total Predictions", f"{display_total:,}", "Live query records", "#3B82F6", '<i class="fa-solid fa-database" style="color: #60A5FA;"></i>')
        with kpi_cols[1]:
            render_kpi("Real News", f"{display_real:,}", f"{real_pct}% of total", "#10B981", '<i class="fa-solid fa-circle-check" style="color: #34D399;"></i>')
        with kpi_cols[2]:
            render_kpi("Fake News", f"{display_fake:,}", f"{fake_pct}% of total", "#EF4444", '<i class="fa-solid fa-triangle-exclamation" style="color: #FB7185;"></i>')
        with kpi_cols[3]:
            render_kpi("Avg. Latency", f"{kpis['avg_inference']}s", "Real-time latency", "#A855F7", '<i class="fa-solid fa-bolt" style="color: #C084FC;"></i>')
        with kpi_cols[4]:
            render_kpi("Best Model", f"{kpis['best_f1']}", f"{kpis['best_model']}", "#3B82F6", '<i class="fa-solid fa-award" style="color: #FBBF24;"></i>')

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        # Middle Row: Model Performance & Confusion Matrix
        col_perf, col_matrix = st.columns([1.3, 1])
        
        with col_perf:
            st.markdown(
                """
                <div class="fna-card">
                    <div class="fna-card-header">
                        <div class="fna-card-title"><i class="fa-solid fa-chart-column" style="color: #818CF8; margin-right: 6px;"></i>Model Performance Comparison</div>
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
                        "Status": "⭐ Top" if m == best_f1_key else "Active"
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
                        <div class="fna-card-title"><i class="fa-solid fa-table-cells" style="color: #38BDF8; margin-right: 6px;"></i>Confusion Matrix & Report</div>
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
                # Arranged cleanly with centered bounds
                cm_col1, cm_col2, cm_col3 = st.columns([0.1, 0.8, 0.1])
                with cm_col2:
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
                        <div class="fna-card-title"><i class="fa-solid fa-cloud" style="color: #60A5FA; margin-right: 6px;"></i>Word Clouds (Text Analysis)</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            col_wc_ctrl1, col_wc_ctrl2 = st.columns([1.5, 1])
            with col_wc_ctrl1:
                wc_type = st.radio("Type", ["Real News", "Fake News"], horizontal=True, label_visibility="collapsed")
            with col_wc_ctrl2:
                if st.button("🔄 Regenerate", key="regen_wc_btn", use_container_width=True, help="Re-compute word clouds from latest raw dataset files"):
                    with st.spinner("Regenerating word clouds from datasets..."):
                        try:
                            generate_wordclouds()
                            st.success("Word clouds updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to update word clouds: {e}")
            
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
                        <div class="fna-card-title"><i class="fa-solid fa-chart-pie" style="color: #C084FC; margin-right: 6px;"></i>Prediction Distribution</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            # Generate Matplotlib Donut Chart matching dark UI
            fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0F172A")
            ax.set_facecolor("#0F172A")
            
            if display_total > 0:
                sizes = [display_real, display_fake]
                colors = ["#10B981", "#EF4444"]
                labels = [f"Real: {real_pct}%", f"Fake: {fake_pct}%"]
                
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
                        <div class="fna-card-title"><i class="fa-solid fa-list-check" style="color: #38BDF8; margin-right: 6px;"></i>Recent Predictions / Review Queue</div>
                    </div>
                """,
                unsafe_allow_html=True
            )
            
            # Only show unverified/pending claims in Recent Predictions / Review Queue
            raw_hist = get_history(limit=50)
            if not raw_hist.empty:
                # Prioritize claims waiting for review
                unverified_hist = raw_hist[raw_hist['status'].astype(str).str.strip().str.lower() != 'admin verified']
                hist_df = unverified_hist.head(8) if not unverified_hist.empty else raw_hist.head(8)
            else:
                hist_df = raw_hist

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
                        <div class="fna-card-title"><i class="fa-solid fa-database" style="color: #60A5FA; margin-right: 6px;"></i>Dataset Summary</div>
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
                        <div class="fna-card-title"><i class="fa-solid fa-arrows-rotate" style="color: #34D399; margin-right: 6px;"></i>Model Retraining</div>
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
            
            if st.button("Start Retraining", use_container_width=True):
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

    elif "Prediction History" in admin_page:
        render_admin_header()
        render_prediction_history_page(is_admin=True)

    elif "Model Comparison" in admin_page:
        render_admin_header()
        render_model_comparison_page()

    elif "Dataset Manager" in admin_page:
        render_admin_header()
        render_dataset_manager_page(is_admin=True)

    elif "Model Training" in admin_page:
        render_admin_header()
        render_model_training_page(is_admin=True)

    elif "Model Insights" in admin_page:
        render_admin_header()
        render_model_insights_page(is_admin=True)

    elif "System Logs" in admin_page:
        render_admin_header()
        render_system_logs_page()

    elif "User Accounts" in admin_page:
        render_admin_header()
        st.markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div style="display: flex; align-items: center; gap: 14px;">
                    <div style="background-color: rgba(99, 102, 241, 0.15); border: 1px solid rgba(99, 102, 241, 0.3); border-radius: 12px; width: 48px; height: 48px; display: flex; align-items: center; justify-content: center; font-size: 1.5rem; color: #818CF8; box-shadow: 0 0 20px rgba(99, 102, 241, 0.25);">
                        <i class="fa-solid fa-users"></i>
                    </div>
                    <div>
                        <h1 style="margin: 0; font-size: 1.85rem; font-weight: 800; color: #F8FAFC;">Registered Users Database</h1>
                        <p style="margin: 2px 0 0 0; color: #94A3B8; font-size: 0.95rem;">
                            Live SQLite database table (<code>USER</code>) holding registered accounts and their verification activity.
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        conn = get_connection()
        query = """
            SELECT 
                u.user_id AS 'User ID',
                u.username AS 'Username',
                u.role AS 'Role',
                COUNT(nq.query_id) AS 'Total Predictions Logged'
            FROM USER u
            LEFT JOIN NEWS_QUERY nq ON u.user_id = nq.user_id
            GROUP BY u.user_id
            ORDER BY u.user_id ASC
        """
        try:
            users_df = pd.read_sql_query(query, conn)
        finally:
            conn.close()

        # KPI Metrics
        u_col1, u_col2, u_col3 = st.columns(3)
        with u_col1:
            render_kpi("Registered Users", f"{len(users_df):,}", "Active user accounts", "#6366F1", '<i class="fa-solid fa-user-group" style="color: #818CF8;"></i>')
        with u_col2:
            admin_count = len(users_df[users_df["Role"].str.lower() == "admin"]) if not users_df.empty else 1
            render_kpi("Administrators", f"{admin_count:,}", "Privileged accounts", "#10B981", '<i class="fa-solid fa-user-shield" style="color: #34D399;"></i>')
        with u_col3:
            user_preds = users_df["Total Predictions Logged"].sum() if not users_df.empty else 0
            render_kpi("User Verifications", f"{user_preds:,}", "Total queries by users", "#38BDF8", '<i class="fa-solid fa-chart-line" style="color: #38BDF8;"></i>')

        st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="fna-card">', unsafe_allow_html=True)
        st.markdown("<h4 style='color: #F8FAFC; margin-top: 0;'><i class='fa-solid fa-table' style='margin-right: 8px; color: #818CF8;'></i>Database Table: <code>USER</code></h4>", unsafe_allow_html=True)
        st.dataframe(users_df, use_container_width=True, hide_index=True)
        
        # Download user list
        csv_data = users_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Registered Users (CSV)",
            data=csv_data,
            file_name="registered_users.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    elif "Settings" in admin_page:
        render_admin_header()
        
        # 1. Admin Password Management Card
        st.markdown(
            """
            <div class="fna-card" style="margin-bottom: 20px;">
                <div class="fna-card-header">
                    <div class="fna-card-title"><i class="fa-solid fa-key" style="color: #F59E0B; margin-right: 8px;"></i>Change Admin Password</div>
                    <span style="font-size: 0.8rem; color: #94A3B8;">Persistent across all devices & restarts</span>
                </div>
            """,
            unsafe_allow_html=True
        )
        with st.form("change_password_form", clear_on_submit=True):
            col_p1, col_p2, col_p3 = st.columns(3)
            with col_p1:
                curr_pwd = st.text_input("Current Password", type="password", placeholder="Enter current password")
            with col_p2:
                new_pwd = st.text_input("New Password", type="password", placeholder="Enter new password (min. 4 chars)")
            with col_p3:
                conf_pwd = st.text_input("Confirm New Password", type="password", placeholder="Re-enter new password")
                
            submitted = st.form_submit_button("Update Password", type="primary")
            if submitted:
                stored_hash = get_admin_password_hash()
                if not curr_pwd or not new_pwd or not conf_pwd:
                    st.error("Please fill in all password fields.")
                elif not verify_password(curr_pwd, stored_hash):
                    st.error("Current password does not match.")
                elif len(new_pwd) < 4:
                    st.warning("New password must be at least 4 characters long.")
                elif new_pwd != conf_pwd:
                    st.error("New password and Confirm password do not match.")
                else:
                    success = update_admin_password(new_pwd)
                    if success:
                        st.success("Admin password successfully updated in database! It is now active on all devices.")
                    else:
                        st.error("Failed to update password in database.")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

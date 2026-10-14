# FakeNewsAI UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Overhaul the user interface and admin panel to match the provided dark-mode designs with KPI cards, prediction history inspector, word cloud toggles, and modern components.

**Architecture:** Inject modern CSS styling via a dedicated `src/ui_theme.py` module. Extend the SQLite database schema in `src/db.py` and query helpers in `src/history.py` to support status tracking, prediction codes, and KPI aggregations. Update `app.py` for the public check-news interface and `pages/admin.py` for the multi-view admin dashboard.

**Tech Stack:** Streamlit, Python, SQLite, Pandas, HTML/CSS, Plotly (for distribution donut chart).

**Spec:** `docs/superpowers/specs/2026-10-13-ui-redesign.md`

## Global Constraints
- Must preserve existing model prediction logic and signatures (`predict()`, `log_query()`, `get_history()`).
- Dark theme styling using colors: Background `#0B1120`/`#0F172A`, Card `#1E293B`, Borders `#334155`, Blue `#2563EB`.
- Responsive layout with Streamlit `wide` page configuration.

---

### Task 1: UI Theme and CSS Module (`src/ui_theme.py`)
**Files:**
- Create: `src/ui_theme.py`
- Test: `tests/test_ui_theme.py`

- [ ] **Step 1: Write test for `src/ui_theme.py`**
- [ ] **Step 2: Run test and verify it fails**
- [ ] **Step 3: Implement `src/ui_theme.py` with custom CSS, dark palette, card components, badge helpers, and KPI card HTML renderers**
- [ ] **Step 4: Run test and verify it passes**

### Task 2: Database Schema & History Enhancements (`src/db.py`, `src/history.py`)
**Files:**
- Modify: `src/db.py`
- Modify: `src/history.py`
- Test: `tests/test_history_enhancements.py`

- [ ] **Step 1: Write tests for KPI calculations, status updates, and filtered history queries**
- [ ] **Step 2: Run tests and verify failure**
- [ ] **Step 3: Add `prediction_code`, `status`, `inference_time` support in `src/db.py` and implement helper functions in `src/history.py`**
- [ ] **Step 4: Run tests and verify they pass**

### Task 3: Redesign Public App (`app.py`)
**Files:**
- Modify: `app.py`

- [ ] **Step 1: Apply custom styling and brand headers in `app.py`**
- [ ] **Step 2: Build the Text Input card with character counter and styled tier selector**
- [ ] **Step 3: Build the Empty State card and Result State card with confidence progress bar**
- [ ] **Step 4: Test running `app.py` and verify syntax/execution**

### Task 4: Redesign Admin Dashboard (`pages/admin.py`)
**Files:**
- Modify: `pages/admin.py`

- [ ] **Step 1: Add top navigation / sub-page router (Overview vs Prediction History)**
- [ ] **Step 2: Build the 6 KPI metric cards, Model Performance table, Confusion Matrix & Classification Report**
- [ ] **Step 3: Build the Word Clouds toggle card, Prediction Distribution donut chart, Recent Queue, Dataset Summary, and Retraining cards**
- [ ] **Step 4: Build the Prediction History sub-page with search, filter controls, data table, and side-panel Prediction Details with "Mark as Reviewed" action**
- [ ] **Step 5: Test running `pages/admin.py` and verify syntax/execution**

### Task 5: End-to-End Verification
- [ ] **Step 1: Run full automated tests with pytest**
- [ ] **Step 2: Verify both pages load cleanly with Streamlit**

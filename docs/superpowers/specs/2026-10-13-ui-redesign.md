# FakeNewsAI User Interface & Admin Dashboard Redesign Spec

## Overview
This specification details the overhaul of the Fake News Detection web application to match the modern dark-themed user interface and functionality shown in the target design screenshots. 

The redesign encompasses two primary surfaces:
1. **Public Application (`app.py`)**:
   - Navigation: Home (Check News) and History.
   - Modern dark UI container with card surfaces, borders, glowing accents, and badges.
   - Text input with real-time character count.
   - Tier selection with descriptive labels.
   - Empty state visual card and rich prediction outcome verdict card.
2. **Admin Dashboard (`pages/admin.py`)**:
   - Navigation: Overview (Dashboard & KPIs) and Prediction History (All predictions & review).
   - Top KPI cards: Total Predictions, Real News, Fake News, Unverified, Avg. Inference Time, Best Model (F1).
   - Model Performance comparison table with "Best 🏆" badge.
   - Confusion matrix heatmap selector and classification report breakdown.
   - Word Clouds text analysis viewer with Real / Fake toggle buttons.
   - Prediction distribution donut chart.
   - Recent predictions / review queue table with action triggers.
   - Dataset summary card & Model retraining control card.
   - Detailed Prediction History with search, multi-filter (Model, Status, Date Range), pagination, and side-panel Prediction Details with "Mark as Reviewed" action.

---

## Architectural Changes & Components

### 1. Centralized UI Theme & Styling (`src/ui_theme.py`)
- Provides `inject_custom_css()` injecting CSS variables and classes:
  - Background palette: Deep slate `#0B1120`, Card background `#0F172A`, Surface `#1E293B`, Border `#334155`.
  - Accent colors: Primary Blue `#2563EB` / `#3B82F6`, Success Green `#10B981`, Danger Red `#EF4444`, Warning Amber `#F59E0B`.
  - Typography: Inter / system sans-serif font stack.
  - Component classes: `.fna-card`, `.fna-badge`, `.fna-kpi`, `.fna-metric`, `.fna-table`, `.fna-progress`.
- Provides helper functions for rendering reusable HTML components (KPI cards, badge pills, status pills).

### 2. Database & Data Model Enhancements (`src/db.py` & `src/history.py`)
- Schema updates:
  - Add `prediction_code` (e.g. `PN-YYYYMMDD-XXX`) to queries/results.
  - Add `status` column to `HISTORY_LOG` / `PREDICTION_RESULT` with values: `'Verified'`, `'Reviewed'`, `'Pending'` (defaulting to `'Reviewed'` or `'Pending'`).
  - Add `inference_time` column to `PREDICTION_RESULT`.
- API additions in `history.py`:
  - `get_kpis()`: Aggregated counts for Total, Real, Fake, Unverified, Avg Inference Time.
  - `get_recent_queue(limit=10)`: Returns structured recent predictions with review status.
  - `update_review_status(query_id, status)`: Sets status to `'Reviewed'`.
  - `get_filtered_history(search=None, model=None, status=None, date_from=None, date_to=None, offset=0, limit=10)`.

### 3. Public Web Application (`app.py`)
- **Brand & Header**: Shield icon `🛡️ FakeNewsAI Detection System` and green status pill `🟢 AI Powered - Using trained models`.
- **Text Input Card**:
  - Live character counter: `Characters: <N> | Supports English and Hindi news`.
  - Tier selection with user-friendly descriptions:
    - `Basic (Naive Bayes - Fast)`
    - `Intermediate (Random Forest / SVM)`
    - `Hardcore (BERT / RoBERTa / LSTM)`
    - `Ensemble (Majority Vote - All Models)`
  - Action button: `🔍 Check News`.
- **Display States**:
  - Empty state: Clean card with icon and prompt to test an article.
  - Result state: Card with verdict pill (`Real News` / `Fake News`), confidence meter, inference latency, and detailed breakdown.

### 4. Admin Dashboard (`pages/admin.py`)
- **Overview Subpage**:
  - 6 KPI stat cards.
  - Model Performance comparison table.
  - Confusion Matrix image viewer & classification report table.
  - Word Clouds viewer with toggle buttons.
  - Prediction Distribution donut chart (Plotly or SVG).
  - Recent Predictions review queue.
  - Dataset summary & retraining trigger.
- **Prediction History Subpage**:
  - Search input, model selector, status selector, and date filters.
  - Interactive table displaying prediction rows.
  - Prediction Details inspector panel on the right with full text, confidence, model info, and "Mark as Reviewed" action button.

---

## Verification & Acceptance Criteria
1. `app.py` runs and displays the modern dark UI, matching Screenshot 1 layout and features.
2. Prediction execution works correctly, showing verdict, confidence score, and updating prediction history.
3. `pages/admin.py` displays both the Overview dashboard (Screenshot 2) and Prediction History review queue (Screenshot 3).
4. All existing model loading and prediction functions continue to work without regression.

import os
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def set_cell_shading(cell, color_hex):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color_hex}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def generate_report_docx():
    doc = Document()

    # Standard 1-inch margins matching the reference PDF layout
    for section in doc.sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # Color palette matching PDF headers and clean professional styling
    blue_header = RGBColor(41, 128, 185)    # #2980b9
    sub_blue = RGBColor(52, 152, 219)       # #3498db
    dark_text = RGBColor(33, 37, 41)        # #212529
    charcoal = RGBColor(51, 51, 51)         # #333333

    # ================= PAGE 1: COVER PAGE =================
    cover_space = doc.add_paragraph()
    cover_space.paragraph_format.space_before = Pt(130)

    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = p_title.add_run("Fake News Detection")
    run_title.font.name = "Calibri"
    run_title.font.size = Pt(36)
    run_title.font.bold = True
    run_title.font.color.rgb = RGBColor(20, 30, 45)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(140)
    run_sub = p_sub.add_run("A Data Science Project Report")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(16)
    run_sub.font.italic = True
    run_sub.font.color.rgb = blue_header

    p_mem_hdr = doc.add_paragraph()
    p_mem_hdr.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_mh = p_mem_hdr.add_run("Project Members")
    r_mh.font.name = "Calibri"
    r_mh.font.size = Pt(15)
    r_mh.font.bold = True
    r_mh.font.underline = True
    r_mh.font.color.rgb = RGBColor(30, 30, 30)

    for member in ["Yadav Mohit", "Sorte Shubham", "Jain Karan"]:
        p_mem = doc.add_paragraph()
        p_mem.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_mem.paragraph_format.space_before = Pt(3)
        p_mem.paragraph_format.space_after = Pt(3)
        r_m = p_mem.add_run(member)
        r_m.font.name = "Calibri"
        r_m.font.size = Pt(14)
        r_m.font.italic = True
        r_m.font.color.rgb = dark_text

    p_doc = doc.add_paragraph()
    p_doc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_doc.paragraph_format.space_before = Pt(90)
    r_doc = p_doc.add_run("Project Documentation\nBachelor of Computer Applications (BCA)\nGujarat Technological University")
    r_doc.font.name = "Calibri"
    r_doc.font.size = Pt(11)
    r_doc.font.color.rgb = RGBColor(120, 130, 140)

    doc.add_page_break()

    # Formatting helpers
    def add_h1(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(16)
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = blue_header
        return p

    def add_h2(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(11)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(13)
        r.font.bold = True
        r.font.color.rgb = sub_blue
        return p

    def add_p(text):
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(6)
        p.paragraph_format.line_spacing = 1.15
        r = p.add_run(text)
        r.font.name = "Calibri"
        r.font.size = Pt(11)
        r.font.color.rgb = charcoal
        return p

    def add_bullet_item(bold_txt, regular_txt):
        p = doc.add_paragraph(style='List Bullet')
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.15
        if bold_txt:
            rb = p.add_run(bold_txt)
            rb.font.name = "Calibri"
            rb.font.size = Pt(11)
            rb.font.bold = True
            rb.font.color.rgb = dark_text
        rt = p.add_run(regular_txt)
        rt.font.name = "Calibri"
        rt.font.size = Pt(11)
        rt.font.color.rgb = charcoal
        return p

    def add_numbered_item(num_str, bold_txt, regular_txt):
        p = doc.add_paragraph()
        p.paragraph_format.left_indent = Inches(0.25)
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.15
        rn = p.add_run(num_str + " ")
        rn.font.name = "Calibri"
        rn.font.size = Pt(11)
        rn.font.bold = True
        rn.font.color.rgb = blue_header
        if bold_txt:
            rb = p.add_run(bold_txt)
            rb.font.name = "Calibri"
            rb.font.size = Pt(11)
            rb.font.bold = True
            rb.font.color.rgb = dark_text
        rt = p.add_run(regular_txt)
        rt.font.name = "Calibri"
        rt.font.size = Pt(11)
        rt.font.color.rgb = charcoal
        return p

    # ================= 1. FUNCTIONALITIES =================
    add_h1("1. Functionalities")
    add_p("This project is built to check whether a piece of news (a headline, article, or social media post) is real or fake. Its main functions are:")
    add_bullet_item("", "Take news text as input (typed directly, pasted from clipboard, or uploaded from a file/URL).")
    add_bullet_item("", "Clean and prepare the text using Natural Language Processing (NLP) so a computer can understand it.")
    add_bullet_item("", "Convert the text into numbers (features) that machine learning and deep learning models can process.")
    add_bullet_item("", "Run the text through a trained model to predict \"Real\" or \"Fake\".")
    add_bullet_item("", "Show the result with a confidence score (e.g., \"92% chance this is Fake\").")
    add_bullet_item("", "Keep a history/log of checked news for future reference, audit trails, and CSV export.")
    add_bullet_item("", "Allow comparison between multiple models (basic, intermediate, and advanced/hardcore) to see which one performs better.")

    # ================= 2. OBJECTIVES =================
    add_h1("2. Objectives")
    add_bullet_item("", "To build a system that automatically detects fake news using data science and machine learning.")
    add_bullet_item("", "To reduce the spread of misinformation by giving people a quick way to verify news.")
    add_bullet_item("", "To compare different machine learning models and find out which one gives the best accuracy.")
    add_bullet_item("", "To learn and apply core data science skills: data cleaning, text processing (NLP), model building, and evaluation.")
    add_bullet_item("", "To create a simple, user-friendly interface so anyone can use it.")
    add_bullet_item("", "To make the project scalable, so it can later work with live news feeds or social media data.")

    # ================= 3. TOOLS, TECHNOLOGIES & MODELS =================
    add_h1("3. Tools, Technologies & Models")
    
    add_h2("Tools")
    add_bullet_item("Jupyter Notebook: ", "For writing, experimenting, and testing code step by step.")
    add_bullet_item("VS Code: ", "For building the full application, modular backend pipelines, and tests.")
    add_bullet_item("GitHub: ", "For version control, repository tracking, and saving project history.")
    add_bullet_item("Streamlit: ", "To build a responsive and interactive web interface.")

    add_h2("Technologies / Libraries")
    add_bullet_item("Python: ", "Main programming language.")
    add_bullet_item("Pandas, NumPy: ", "For handling, filtering, and cleaning structured datasets.")
    add_bullet_item("NLTK: ", "For natural language processing (removing stopwords, regex cleaning, stemming, lemmatization).")
    add_bullet_item("Scikit-learn: ", "For TF-IDF vectorization, traditional machine learning models, and evaluation metrics.")
    add_bullet_item("PyTorch / Transformers: ", "For deep learning sequence models and BERT transformer architectures.")
    add_bullet_item("Matplotlib / Seaborn: ", "For generating graphs, confusion matrices, and visualizations.")
    add_bullet_item("SQLite: ", "For persistent database storage of queries and prediction history.")

    add_h2("Models (Basic - Intermediate - Hardcore)")
    add_p("(Models structured by operational tiers according to the project implementation):")
    add_bullet_item("Naive Bayes: ", "Probabilistic classifier leveraging word frequencies (95.55% accuracy).")
    add_bullet_item("Logistic Regression: ", "Linear classification boundary with L2 regularization (99.26% accuracy).")
    add_bullet_item("Decision Tree: ", "Rule-based non-linear tree partitioning (99.61% accuracy).")
    add_bullet_item("Random Forest: ", "Ensemble bagging classifier with randomized decision trees (99.63% accuracy).")
    add_bullet_item("Support Vector Machine (SVM): ", "Linear maximum-margin hyperplane separator (99.41% accuracy).")
    add_bullet_item("LSTM (Long Short-Term Memory): ", "Recurrent Neural Network modeling sequential text dependencies.")
    add_bullet_item("BERT (Bidirectional Encoder Representations from Transformers): ", "Contextual transformer architecture capturing bidirectional semantic dependencies (99.26% accuracy).")

    # ================= 4. MAIN FEATURES =================
    add_h1("4. Main Features")
    add_bullet_item("Text Input Box: ", "Paste any news headline or article content directly.")
    add_bullet_item("Real-time Prediction: ", "Get result instantly after clicking \"Check News\".")
    add_bullet_item("Confidence Score: ", "Shows how sure the model is about its answer (e.g., 99.2% Fake).")
    add_bullet_item("Multiple Model Options: ", "User can pick which model tier (Basic, Intermediate, or Hardcore) to use.")
    add_bullet_item("Visualization Dashboard: ", "Shows accuracy, confusion matrix, and word clouds of fake vs real news.")
    add_bullet_item("History Log: ", "Stores previously checked news articles with timestamps and CSV export functionality.")
    add_bullet_item("Simple UI: ", "Clean and minimal design with modern typography and theme support so it is easy for anyone to use.")

    # ================= 5. MODULES =================
    add_h1("5. Modules")
    add_numbered_item("1.", "Data Collection Module: ", "Gathers news datasets (from Kaggle True/Fake news corpus, FakeNewsNet, etc.).")
    add_numbered_item("2.", "Data Preprocessing Module: ", "Cleans text (removes punctuation, URLs, stopwords, converts to lowercase, tokenizes).")
    add_numbered_item("3.", "Feature Extraction Module: ", "Converts text into numeric form using TF-IDF, Bag of Words, or Word Embeddings.")
    add_numbered_item("4.", "Model Training Module: ", "Trains the chosen ML/DL model on the prepared data and serializes artifacts into models/.")
    add_numbered_item("5.", "Model Evaluation Module: ", "Tests accuracy, precision, recall, F1-score, and outputs confusion matrix visuals.")
    add_numbered_item("6.", "Prediction Module: ", "Takes new/unseen news and gives a Real/Fake result with confidence scoring.")
    add_numbered_item("7.", "User Interface Module: ", "The front-end Streamlit web interface where users interact with the system.")
    add_numbered_item("8.", "Result Storage Module: ", "Saves results and logs to SQLite database for auditing and future use.")

    # ================= 6. HOW IT WORKS =================
    add_h1("6. How It Works (Step-by-Step)")
    add_numbered_item("1.", "", "User enters a news headline or article into the system.")
    add_numbered_item("2.", "", "The text is cleaned (removing extra spaces, special characters, and stopwords like \"the\", \"is\", \"and\").")
    add_numbered_item("3.", "", "The cleaned text is converted into numbers using a technique like TF-IDF or word embeddings.")
    add_numbered_item("4.", "", "This numeric data is fed into the trained machine learning or deep learning model.")
    add_numbered_item("5.", "", "The model analyzes patterns it learned during training (from thousands of real and fake news examples).")
    add_numbered_item("6.", "", "It predicts whether the news is \"Real\" or \"Fake\" along with a confidence percentage.")
    add_numbered_item("7.", "", "The result is displayed to the user on the screen and archived in the database log.")

    # ================= 7. LIFE CYCLE =================
    add_h1("7. Life Cycle of the Project")
    add_numbered_item("1.", "Problem Identification: ", "Understand what \"fake news\" means and why detecting it matters.")
    add_numbered_item("2.", "Data Collection: ", "Gather a labeled dataset (news marked as real or fake).")
    add_numbered_item("3.", "Data Cleaning & Preprocessing: ", "Remove noise, missing values, duplicates, and unwanted symbols.")
    add_numbered_item("4.", "Exploratory Data Analysis (EDA): ", "Study the data using graphs and stats to find patterns and word distributions.")
    add_numbered_item("5.", "Feature Engineering: ", "Convert text into machine-readable numerical feature representations.")
    add_numbered_item("6.", "Model Selection & Training: ", "Choose and train basic, intermediate, and hardcore models.")
    add_numbered_item("7.", "Model Evaluation: ", "Test how well each model performs using accuracy, precision, recall, and F1-score.")
    add_numbered_item("8.", "Model Deployment: ", "Put the best model into a working Streamlit web application.")
    add_numbered_item("9.", "Testing: ", "Check the whole system with unit test suites and real-world news articles.")
    add_numbered_item("10.", "Maintenance & Updates: ", "Keep improving and retraining the model as new fake news patterns appear.")

    # ================= 8. ADVANTAGES & DISADVANTAGES =================
    add_h1("8. Advantages and Disadvantages")
    
    add_h2("Advantages")
    add_bullet_item("", "Helps stop the spread of misinformation quickly.")
    add_bullet_item("", "Saves time compared to manual fact-checking.")
    add_bullet_item("", "Can be used by news agencies, students, researchers, and everyday readers.")
    add_bullet_item("", "Once trained, it can process huge amounts of news very fast (milliseconds per article).")
    add_bullet_item("", "Multiple models give flexibility — light and fast on CPU, or heavy and highly accurate using BERT.")

    add_h2("Disadvantages")
    add_bullet_item("", "Accuracy depends heavily on the quality and balance of training data.")
    add_bullet_item("", "May struggle with sarcasm, satire (e.g., The Onion), or completely new/unseen topics.")
    add_bullet_item("", "Advanced models (like BERT) need strong hardware (GPU/RAM) to train.")
    add_bullet_item("", "Cannot fully replace human judgment, ground-truth fact verification, and critical thinking.")
    add_bullet_item("", "Needs regular updates and retraining, since fake news linguistic patterns keep changing over time.")

    # ================= 9. OVERVIEW =================
    add_h1("9. Overview")
    add_p(
        "Fake News Detection is a data science project that uses Natural Language Processing (NLP) "
        "and Machine Learning to automatically classify news content as real or fake. It studies patterns "
        "in language, word usage, and writing style that are common in misleading news, and uses this "
        "knowledge to make predictions on new articles. The project is built at different complexity levels — "
        "from simple models like Logistic Regression and Naive Bayes to advanced deep learning models like "
        "BERT — making it a comprehensive project to demonstrate a complete data science pipeline."
    )

    # ================= 10. WHY IT IS USEFUL =================
    add_h1("10. Why It Is Useful")
    add_bullet_item("", "Misinformation spreads fast on social media, and manual checking is slow and hard to scale.")
    add_bullet_item("", "It gives an extra layer of protection for readers who can't verify every source themselves.")
    add_bullet_item("", "It supports journalists, researchers, and fact-checking organizations by pre-filtering suspicious content.")
    add_bullet_item("", "It builds public awareness about how fake news is written and structured.")
    add_bullet_item("", "As a college project, it shows practical, real-world use of data science and AI skills.")

    # ================= 11. HOW TO USE IT =================
    add_h1("11. How to Use It")
    add_numbered_item("1.", "", "Open the web app / interface by launching `streamlit run app.py`.")
    add_numbered_item("2.", "", "Paste or type the news headline/article you want to check.")
    add_numbered_item("3.", "", "Select a model tier (Basic, Intermediate, or Hardcore).")
    add_numbered_item("4.", "", "Click the \"Check News\" button.")
    add_numbered_item("5.", "", "View the result: \"Real News\" or \"Fake News\", along with the confidence score.")
    add_numbered_item("6.", "", "Optionally, check the history tab to see previously analyzed news and download records.")

    # ================= 12. SCOPE =================
    add_h1("12. Scope")
    add_bullet_item("", "Can be expanded to detect fake news in multiple languages using multilingual transformer embeddings.")
    add_bullet_item("", "Can be connected to live news APIs, RSS feeds, or social media data for real-time detection.")
    add_bullet_item("", "Can be turned into a browser extension that checks news while users browse social media.")
    add_bullet_item("", "Can be used by schools, media houses, government agencies, and fact-checking organizations.")
    add_bullet_item("", "Can be integrated with chatbots (e.g., Telegram/WhatsApp bots) to instantly answer \"Is this news true?\".")

    # ================= 13. SYSTEM ARCHITECTURE =================
    add_h1("13. System Architecture")

    # Diagram Table matching PDF layout
    table_diag = doc.add_table(rows=1, cols=1)
    table_diag.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table_diag.cell(0, 0)
    set_cell_shading(cell, "F7F9FA")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=200)

    diag_content = (
        "User Input (News Text)\n"
        "         │\n"
        "         ▼\n"
        "Data Preprocessing (Cleaning, Tokenizing)\n"
        "         │\n"
        "         ▼\n"
        "Feature Extraction (TF-IDF / Word Embeddings)\n"
        "         │\n"
        "         ▼\n"
        "Trained ML/DL Model (Basic / Intermediate / Hardcore)\n"
        "         │\n"
        "         ▼\n"
        "Prediction Engine (Real / Fake + Confidence Score)\n"
        "         │\n"
        "         ├─────────────────────────────────────────┐\n"
        "         ▼                                         ▼\n"
        "User Interface (Result Displayed to User)      Database/Log (Stores History of Checked News)"
    )
    p_box = cell.paragraphs[0]
    p_box.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_diag = p_box.add_run(diag_content)
    r_diag.font.name = "Courier New"
    r_diag.font.size = Pt(9.5)
    r_diag.font.color.rgb = RGBColor(40, 55, 70)

    add_p("")
    p_fe = doc.add_paragraph()
    p_fe.paragraph_format.space_before = Pt(6)
    p_fe.paragraph_format.space_after = Pt(2)
    r_fe = p_fe.add_run("Flow explanation:")
    r_fe.font.name = "Calibri"
    r_fe.font.bold = True
    r_fe.font.color.rgb = dark_text

    add_p(
        "The user gives input, the system cleans it, converts it into numeric features, passes it to the "
        "trained model, the model predicts the result, and the result is shown back to the user and saved in the log."
    )

    # ================= 14. HARDWARE & SOFTWARE REQUIREMENTS =================
    add_h1("14. Software Requirements & Hardware Requirements")

    add_h2("Hardware Requirements")
    add_bullet_item("Processor: ", "Intel i3 or above (i5/i7 recommended for deep learning models).")
    add_bullet_item("RAM: ", "Minimum 4GB (8GB+ recommended for hardcore models like BERT).")
    add_bullet_item("Storage: ", "At least 2GB free space.")
    add_bullet_item("GPU: ", "Optional but recommended for deep learning models.")

    add_h2("Software Requirements")
    add_bullet_item("Operating System: ", "Windows / macOS / Linux.")
    add_bullet_item("Python Version: ", "Python 3.8 or above.")
    add_bullet_item("IDE / Tools: ", "Jupyter Notebook / VS Code, Git.")
    add_bullet_item("Required Python Libraries: ", "Pandas, NumPy, Scikit-learn, NLTK, PyTorch / Transformers, Matplotlib / Seaborn, Streamlit (for UI).")
    add_bullet_item("Internet Connection: ", "For downloading datasets and pretrained transformer libraries.")

    # Benchmark Summary Table
    add_h2("Evaluated Model Benchmark Summary")
    perf_data = [
        ("Tier", "Model", "Accuracy", "Precision", "Recall", "F1 Score"),
        ("Basic", "Naive Bayes", "95.55%", "95.68%", "95.81%", "95.74%"),
        ("Basic", "Decision Tree", "99.61%", "99.81%", "99.43%", "99.62%"),
        ("Basic", "Logistic Regression", "99.26%", "99.32%", "99.28%", "99.30%"),
        ("Intermediate", "Random Forest", "99.63%", "99.77%", "99.52%", "99.64%"),
        ("Intermediate", "SVM", "99.41%", "99.42%", "99.45%", "99.43%"),
        ("Hardcore", "BERT", "99.26%", "99.29%", "99.29%", "99.29%"),
    ]

    tbl = doc.add_table(rows=len(perf_data), cols=6)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False

    col_widths = [Inches(1.1), Inches(1.8), Inches(1.0), Inches(1.0), Inches(1.0), Inches(1.0)]

    for row_idx, row_values in enumerate(perf_data):
        row = tbl.rows[row_idx]
        is_hdr = (row_idx == 0)
        for col_idx, val in enumerate(row_values):
            c = row.cells[col_idx]
            c.width = col_widths[col_idx]
            set_cell_margins(c, top=80, bottom=80, left=100, right=100)
            
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if col_idx >= 2 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.name = "Calibri"
            r.font.size = Pt(10)

            if is_hdr:
                set_cell_shading(c, "2980B9")
                r.font.bold = True
                r.font.color.rgb = RGBColor(255, 255, 255)
            else:
                set_cell_shading(c, "F9FAFC" if row_idx % 2 == 1 else "FFFFFF")
                r.font.color.rgb = charcoal

    add_p("")

    # ================= CONCLUSION =================
    add_h1("Conclusion")
    add_p(
        "This Fake News Detection project is a great example of how data science and AI can solve a real-world "
        "problem. By combining data cleaning, NLP, and machine learning (from basic to advanced models), it shows "
        "a complete data science pipeline that is perfect for a college submission — simple enough to explain, yet "
        "advanced enough to demonstrate strong technical skills."
    )

    out_file = r"c:\Users\ymohi\fake_news_detection\docs\PROJECT_DOCUMENTATION.docx"
    doc.save(out_file)
    print(f"File created successfully at: {out_file}")

if __name__ == "__main__":
    generate_report_docx()

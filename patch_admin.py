import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """        st.subheader("Model Performance")"""

replacement = """        col_title, col_logout = st.columns([8, 2])
        with col_logout:
            if st.button("Logout"):
                st.session_state.admin_authenticated = False
                st.rerun()

        st.subheader("Dataset Statistics")
        true_path = os.path.join(BASE_DIR, "data", "raw", "True.csv")
        fake_path = os.path.join(BASE_DIR, "data", "raw", "Fake.csv")
        try:
            true_count = len(pd.read_csv(true_path))
            fake_count = len(pd.read_csv(fake_path))
        except:
            true_count, fake_count = 0, 0
            
        col_ds1, col_ds2, col_ds3 = st.columns(3)
        col_ds1.metric("Real News Records", true_count)
        col_ds2.metric("Fake News Records", fake_count)
        col_ds3.metric("Total Records", true_count + fake_count)
        
        st.subheader("Database Management")
        if st.button("Clear Prediction History", type="primary"):
            import sqlite3
            conn = sqlite3.connect(os.path.join(BASE_DIR, "data", "history.db"))
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions")
            conn.commit()
            conn.close()
            st.success("History cleared successfully!")
            
        st.subheader("Model Performance")"""

if target in content:
    content = content.replace(target, replacement)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully added new features to Admin Dashboard in app.py")
else:
    print("Target not found in app.py")

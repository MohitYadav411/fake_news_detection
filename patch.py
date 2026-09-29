import os
import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """        st.subheader("Confusion Matrices")
        matrix_model = st.selectbox("Select model to view confusion matrix", [m["model"] for m in all_metrics] if all_metrics else [])
        if matrix_model:
            img_path = os.path.join(BASE_DIR, "models", f"{matrix_model}_confusion_matrix.png")
            if os.path.exists(img_path):
                st.image(img_path, caption=f"{matrix_model} Confusion Matrix")
            else:
                st.info(f"No confusion matrix image found for {matrix_model}.")"""

replacement = target + """

        st.subheader("Word Clouds (Fake vs Real News)")
        col_wc1, col_wc2 = st.columns(2)
        with col_wc1:
            true_wc_path = os.path.join(BASE_DIR, "models", "wordcloud_true.png")
            if os.path.exists(true_wc_path):
                st.image(true_wc_path, caption="Real News Word Cloud", use_container_width=True)
            else:
                st.info("True news word cloud not found.")
        with col_wc2:
            fake_wc_path = os.path.join(BASE_DIR, "models", "wordcloud_fake.png")
            if os.path.exists(fake_wc_path):
                st.image(fake_wc_path, caption="Fake News Word Cloud", use_container_width=True)
            else:
                st.info("Fake news word cloud not found.")"""

if target in content:
    content = content.replace(target, replacement)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully updated app.py")
else:
    print("Target not found.")

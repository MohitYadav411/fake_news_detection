import os

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target1 = """MODEL_TIERS = {
    "Basic": ["naive_bayes", "decision_tree", "logistic_regression"],
    "Intermediate": ["random_forest", "svm"],
    "Hardcore": ["lstm", "bert"]
}"""

replacement1 = """MODEL_TIERS = {
    "Basic": ["naive_bayes", "decision_tree", "logistic_regression"],
    "Intermediate": ["random_forest", "svm"],
    "Hardcore": ["lstm", "bert"],
    "Ensemble (Majority Vote)": ["naive_bayes", "decision_tree", "logistic_regression", "random_forest", "svm", "lstm", "bert"]
}"""

target2 = """                st.subheader(f"Results for {tier} Tier")
                cols = st.columns(len(models_to_run))
                for i, m_name in enumerate(models_to_run):
                    with cols[i]:
                        st.write(f"**{m_name.replace('_', ' ').title()}**")
                        with st.spinner("Analyzing..."):
                            result = predict(raw_text, m_name)
                            if "error" in result:
                                st.error(f"Error: {result['error']}")
                            else:
                                label = result["label"]
                                confidence = result["confidence"]
                                if label == "Real":
                                    st.success(f"Prediction: {label}")
                                else:
                                    st.error(f"Prediction: {label}")
                                st.progress(min(int(confidence) / 100.0, 1.0))
                                st.write(f"Confidence: {confidence}%")
                                # Log query
                                cleaned_text = preprocess(raw_text)
                                log_query(raw_text, cleaned_text, m_name, label, confidence)
                st.caption("Caveat: This is a decision-support tool, not a final verdict. Always verify facts independently.")"""

replacement2 = """                st.subheader(f"Results for {tier} Tier")
                
                if tier == "Ensemble (Majority Vote)":
                    real_votes = 0
                    fake_votes = 0
                    results_data = []
                    
                    with st.spinner("Running all 7 models concurrently..."):
                        for m_name in models_to_run:
                            result = predict(raw_text, m_name)
                            if "error" not in result:
                                label = result["label"]
                                if label == "Real":
                                    real_votes += 1
                                else:
                                    fake_votes += 1
                                results_data.append({"Model": m_name.replace('_', ' ').title(), "Prediction": label, "Confidence": f"{result['confidence']}%"})
                                
                                # Log query
                                cleaned_text = preprocess(raw_text)
                                log_query(raw_text, cleaned_text, m_name, label, result["confidence"])
                    
                    # Final verdict
                    if real_votes > fake_votes:
                        st.success(f"### 🎯 FINAL VERDICT: REAL NEWS ({real_votes} out of {len(models_to_run)} models agree)")
                    else:
                        st.error(f"### 🛑 FINAL VERDICT: FAKE NEWS ({fake_votes} out of {len(models_to_run)} models agree)")
                        
                    st.write("#### Detailed Breakdown:")
                    st.dataframe(results_data, use_container_width=True)
                    
                else:
                    cols = st.columns(len(models_to_run))
                    for i, m_name in enumerate(models_to_run):
                        with cols[i]:
                            st.write(f"**{m_name.replace('_', ' ').title()}**")
                            with st.spinner("Analyzing..."):
                                result = predict(raw_text, m_name)
                                if "error" in result:
                                    st.error(f"Error: {result['error']}")
                                else:
                                    label = result["label"]
                                    confidence = result["confidence"]
                                    if label == "Real":
                                        st.success(f"Prediction: {label}")
                                    else:
                                        st.error(f"Prediction: {label}")
                                    st.progress(min(int(confidence) / 100.0, 1.0))
                                    st.write(f"Confidence: {confidence}%")
                                    # Log query
                                    cleaned_text = preprocess(raw_text)
                                    log_query(raw_text, cleaned_text, m_name, label, confidence)
                st.caption("Caveat: This is a decision-support tool, not a final verdict. Always verify facts independently.")"""

if target1 in content and target2 in content:
    content = content.replace(target1, replacement1)
    content = content.replace(target2, replacement2)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched app.py for Ensemble mode")
else:
    print("Target not found")

import sys

with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

target = """        tier = st.selectbox("Model Tier", list(MODEL_TIERS.keys()))
        tier_default_models = {
            "Basic": "logistic_regression",
            "Intermediate": "random_forest",
            "Hardcore": "bert"
        }
        model_name = tier_default_models[tier]

        check_button = st.button("Check News", disabled=len(raw_text.strip()) == 0)

        if check_button:
            if not raw_text.strip():
                st.warning("Please enter some text to check.")
            else:
                with st.spinner(f"Analyzing with {model_name}..."):
                    result = predict(raw_text, model_name)
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        label = result["label"]
                        confidence = result["confidence"]
                        
                        # Display Result
                        st.subheader("Analysis Result")
                        if label == "Real":
                            st.success(f"**Prediction: {label}**")
                        else:
                            st.error(f"**Prediction: {label}**")
                            
                        # Use min to ensure we don't pass > 1.0 to st.progress
                        st.progress(min(int(confidence) / 100.0, 1.0))
                        st.write(f"**Confidence Score:** {confidence}%")
                        st.caption("Caveat: This is a decision-support tool, not a final verdict. Always verify facts independently.")

                        # Log query
                        cleaned_text = preprocess(raw_text)
                        log_query(raw_text, cleaned_text, model_name, label, confidence)"""

replacement = """        tier = st.selectbox("Model Tier", list(MODEL_TIERS.keys()))
        models_to_run = MODEL_TIERS[tier]

        check_button = st.button("Check News", disabled=len(raw_text.strip()) == 0)

        if check_button:
            if not raw_text.strip():
                st.warning("Please enter some text to check.")
            else:
                st.subheader(f"Results for {tier} Tier")
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

if target in content:
    content = content.replace(target, replacement)
    with open('app.py', 'w', encoding='utf-8') as f:
        f.write(content)
    print("Successfully patched app.py")
else:
    print("Target not found")

import sys
import os

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from predict import predict

def run_sanity_checks():
    # 1. Obviously Real
    real_text = "The Federal Reserve today announced a new policy regarding interest rates, raising the benchmark by 0.25 percentage points to combat inflation."
    # 2. Obviously Fake
    fake_text = "Alien spacecraft found hidden in the basement of the White House! The president confirms communication with extraterrestrials."
    # 3. Satire / Borderline
    satire_text = "Local man literally too angry to die after waiting in line at the DMV for four hours."
    
    print("--- SANITY CHECK ---")
    
    for model_name in ["naive_bayes", "logistic_regression", "random_forest"]:
        print(f"\nModel: {model_name}")
        
        print("\nObviously Real:")
        result = predict(real_text, model_name)
        print(result)
        
        print("\nObviously Fake:")
        result = predict(fake_text, model_name)
        print(result)
        
        print("\nSatire/Borderline:")
        result = predict(satire_text, model_name)
        print(result)

if __name__ == '__main__':
    run_sanity_checks()

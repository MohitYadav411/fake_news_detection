import sys
import os
import unittest

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict

class TestPredict(unittest.TestCase):
    
    def test_empty_string(self):
        result = predict("", "naive_bayes")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Please enter some text to check")
        
    def test_whitespace_only(self):
        result = predict("   \n \t ", "naive_bayes")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Please enter some text to check")
        
    def test_cleaned_to_empty(self):
        # Stopwords only
        result = predict("is the and of it", "naive_bayes")
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Not enough content to analyze")
        
    def test_extremely_long_input(self):
        # Generate string of 5005 words
        long_text = "word " * 5005
        # The function should process the first 5000 and return a valid prediction dict
        # Assuming the model returns valid keys 'label' and 'confidence'
        result = predict(long_text, "naive_bayes")
        self.assertNotIn("error", result)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        
    def test_missing_model(self):
        result = predict("some valid news text", "nonexistent_model")
        self.assertIn("error", result)
        self.assertTrue("hasn't been trained yet" in result["error"])

if __name__ == '__main__':
    unittest.main()

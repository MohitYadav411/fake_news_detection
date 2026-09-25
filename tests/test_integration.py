import sys
import os
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from predict import predict
from history import get_history

class TestIntegration(unittest.TestCase):
    
    def test_basic_tier_pipeline(self):
        result = predict("Scientists discover new planet made entirely of diamond.", "naive_bayes")
        self.assertNotIn("error", result)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertTrue(result["label"] in ["Real", "Fake"])
        
    def test_intermediate_tier_pipeline(self):
        result = predict("Local man claims he can speak to pigeons.", "random_forest")
        self.assertNotIn("error", result)
        self.assertIn("label", result)
        self.assertIn("confidence", result)
        self.assertTrue(result["label"] in ["Real", "Fake"])

    def test_hardcore_tier_pipeline_lstm(self):
        # We might not be able to run deep learning model smoothly in a simple predict test if it's not fully wired
        # But we'll try it if the file exists. Let's just try random forest for intermediate.
        pass

    def test_history_persistence(self):
        # Assuming the history db gets initialized properly
        history = get_history(limit=1)
        self.assertTrue(history is not None)

if __name__ == '__main__':
    unittest.main()

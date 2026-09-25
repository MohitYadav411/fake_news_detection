import unittest
import os
import sys

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features import fit_vectorizer, save_vectorizer, load_vectorizer, transform

class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "fake news article here",
            "real news is completely different",
            "some more fake news"
        ]
        self.vectorizer_path = "test_vectorizer.joblib"

    def tearDown(self):
        if os.path.exists(self.vectorizer_path):
            os.remove(self.vectorizer_path)

    def test_fit_and_transform(self):
        vec = fit_vectorizer(self.corpus, max_features=10)
        self.assertIsNotNone(vec)
        
        # Test transform
        matrix = transform(vec, ["fake news"])
        self.assertEqual(matrix.shape[1], len(vec.vocabulary_))
        self.assertTrue(matrix.shape[0] == 1)

    def test_save_and_load(self):
        vec = fit_vectorizer(self.corpus, max_features=10)
        save_vectorizer(vec, self.vectorizer_path)
        
        self.assertTrue(os.path.exists(self.vectorizer_path))
        
        loaded_vec = load_vectorizer(self.vectorizer_path)
        self.assertEqual(vec.vocabulary_, loaded_vec.vocabulary_)

if __name__ == '__main__':
    unittest.main()

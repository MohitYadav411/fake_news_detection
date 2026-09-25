import unittest
import sys
import os

# Ensure src can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import clean_text, tokenize, remove_stopwords, lemmatize, preprocess

class TestPreprocessing(unittest.TestCase):
    def test_clean_text_basic(self):
        text = "Hello WORLD!"
        self.assertEqual(clean_text(text), "hello world")

    def test_clean_text_html(self):
        text = "This is <b>bold</b> and <a href='link'>HTML</a>."
        self.assertEqual(clean_text(text), "this is bold and html")

    def test_clean_text_url(self):
        text = "Check out http://example.com and https://google.com/search?q=fake+news!"
        self.assertEqual(clean_text(text), "check out and")

    def test_tokenize(self):
        text = "hello world this is a test"
        self.assertEqual(tokenize(text), ["hello", "world", "this", "is", "a", "test"])

    def test_remove_stopwords(self):
        tokens = ["this", "is", "a", "fake", "news", "article"]
        self.assertEqual(remove_stopwords(tokens), ["fake", "news", "article"])

    def test_lemmatize(self):
        tokens = ["running", "cats", "better"]
        # 'running' lemmas to 'running' by default without pos tag, 'cats' -> 'cat'
        self.assertEqual(lemmatize(tokens), ["running", "cat", "better"])

    def test_preprocess_empty_and_stopwords(self):
        # Only stop words
        self.assertEqual(preprocess("this is a the and or"), "insufficient content to analyze")
        # Empty string
        self.assertEqual(preprocess(""), "insufficient content to analyze")
        # Only punctuation
        self.assertEqual(preprocess("!!! ??? ..."), "insufficient content to analyze")

    def test_preprocess_full_pipeline(self):
        raw = "BREAKING NEWS: Cats are running http://cats.com <br>!!!"
        # 'breaking news cats are running'
        # stopwords removed: 'breaking', 'news', 'cats', 'running'
        # lemmatized: 'breaking', 'news', 'cat', 'running'
        self.assertEqual(preprocess(raw), "breaking news cat running")
        
    def test_preprocess_non_ascii(self):
        raw = "Café au lait is great ñ"
        # 'café', 'au', 'lait', 'great', 'ñ'
        self.assertEqual(preprocess(raw), "café au lait great ñ")

if __name__ == '__main__':
    unittest.main()

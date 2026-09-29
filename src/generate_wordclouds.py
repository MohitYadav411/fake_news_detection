import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_wordclouds():
    print("Generating Word Clouds...")
    
    true_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'True.csv'))
    fake_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'raw', 'Fake.csv'))
    
    # We will sample 2000 articles from each to speed up wordcloud generation
    true_text = " ".join(true_df['text'].dropna().sample(min(2000, len(true_df)), random_state=42).tolist())
    fake_text = " ".join(fake_df['text'].dropna().sample(min(2000, len(fake_df)), random_state=42).tolist())
    
    print("Generating True News Word Cloud...")
    wc_true = WordCloud(width=800, height=400, background_color='white').generate(true_text)
    wc_true.to_file(os.path.join(BASE_DIR, 'models', 'wordcloud_true.png'))
    
    print("Generating Fake News Word Cloud...")
    wc_fake = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(fake_text)
    wc_fake.to_file(os.path.join(BASE_DIR, 'models', 'wordcloud_fake.png'))
    
    print("Word clouds saved to models/ directory.")

if __name__ == "__main__":
    generate_wordclouds()

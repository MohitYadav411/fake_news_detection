import pandas as pd
from datasets import load_dataset
import os

print('Downloading FakeNewsNet...')
dataset = load_dataset('Ahren09/FakeNewsNet', split='train')
df = pd.DataFrame(dataset)

# Determine the column names. Typically they are 'title', 'text' (or 'news_text'), 'label'
print(df.columns)
if 'news_url' in df.columns:
    df = df.rename(columns={'news_url': 'url'})

# In FakeNewsNet, label 0 is usually True and label 1 is usually Fake.
# Let's see if we can deduce based on common convention, or just assume 0=Real, 1=Fake.
hf_true = df[df['label'] == 0].copy()
hf_fake = df[df['label'] == 1].copy()

# Just in case columns are different
if 'text' not in df.columns:
    if 'news_text' in df.columns:
        hf_true['text'] = hf_true['news_text']
        hf_fake['text'] = hf_fake['news_text']
    elif 'tweet' in df.columns:
        hf_true['text'] = hf_true['tweet']
        hf_fake['text'] = hf_fake['tweet']
    else:
        hf_true['text'] = ''
        hf_fake['text'] = ''

if 'title' not in df.columns:
    hf_true['title'] = ''
    hf_fake['title'] = ''

hf_true = hf_true[['title', 'text']].dropna()
hf_fake = hf_fake[['title', 'text']].dropna()

existing_true = pd.read_csv('data/raw/True.csv')
existing_fake = pd.read_csv('data/raw/Fake.csv')

final_true = pd.concat([existing_true, hf_true], ignore_index=True).drop_duplicates(subset=['text'])
final_fake = pd.concat([existing_fake, hf_fake], ignore_index=True).drop_duplicates(subset=['text'])

final_true.to_csv('data/raw/True.csv', index=False)
final_fake.to_csv('data/raw/Fake.csv', index=False)

print(f"Added {len(hf_true)} real news and {len(hf_fake)} fake news to the dataset.")
print(f"Total True: {len(final_true)}, Total Fake: {len(final_fake)}")

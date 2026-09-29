import pandas as pd
from datasets import load_dataset
import random

print("Loading mrm8488/fake-news from HuggingFace...")
dataset = load_dataset('mrm8488/fake-news', split='train')
df_hf = pd.DataFrame(dataset)

# mrm8488/fake-news: label 0 = True, label 1 = Fake
hf_true = df_hf[df_hf['label'] == 0].sample(5000, random_state=42)
hf_fake = df_hf[df_hf['label'] == 1].sample(5000, random_state=42)

# existing data
existing_true = pd.read_csv('data/raw/True.csv')
existing_fake = pd.read_csv('data/raw/Fake.csv')

print(f"Existing True: {len(existing_true)}, Existing Fake: {len(existing_fake)}")

# ensure title column exists for HF
hf_true['title'] = ''
hf_fake['title'] = ''

# reorder
hf_true = hf_true[['title', 'text']]
hf_fake = hf_fake[['title', 'text']]

# Combine
final_true = pd.concat([existing_true, hf_true], ignore_index=True)
final_fake = pd.concat([existing_fake, hf_fake], ignore_index=True)

# Drop duplicates just in case
final_true = final_true.drop_duplicates(subset=['text'])
final_fake = final_fake.drop_duplicates(subset=['text'])

# Save
final_true.to_csv('data/raw/True.csv', index=False)
final_fake.to_csv('data/raw/Fake.csv', index=False)

print(f"Generated robust dataset: {len(final_true)} True, {len(final_fake)} Fake.")

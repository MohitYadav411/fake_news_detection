import pandas as pd
from datasets import load_dataset
import os

print("Loading BBC News dataset...")
ds_train = load_dataset('SetFit/bbc-news', split='train')
ds_test = load_dataset('SetFit/bbc-news', split='test')

# Convert to pandas
df_train = pd.DataFrame(ds_train)
df_test = pd.DataFrame(ds_test)

# Combine splits
df_bbc = pd.concat([df_train, df_test], ignore_index=True)

# The bbc dataset has 'text' column. We will set 'title' to empty.
df_bbc['title'] = ''
df_bbc = df_bbc[['title', 'text']]

print(f"Loaded {len(df_bbc)} BBC News articles (Real News).")

# Append to True.csv
true_path = os.path.join('data', 'raw', 'True.csv')
existing_true = pd.read_csv(true_path)

final_true = pd.concat([existing_true, df_bbc], ignore_index=True)
final_true = final_true.drop_duplicates(subset=['text'])

final_true.to_csv(true_path, index=False)
print(f"Updated True.csv. Total Real News records now: {len(final_true)}")

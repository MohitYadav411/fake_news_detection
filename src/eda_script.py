import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nbformat as nbf

# Read datasets
print("Loading datasets...")
fake_df = pd.read_csv('data/raw/Fake.csv')
true_df = pd.read_csv('data/raw/True.csv')

# Add labels
fake_df['label'] = 'fake'
true_df['label'] = 'real'

# Combine
df = pd.concat([fake_df, true_df], ignore_index=True)

# 8.1 Class balance
print("Class balance:")
balance = df['label'].value_counts()
print(balance)

# 8.2 Text length
df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))

# Save plots
os.makedirs('notebooks/plots', exist_ok=True)

plt.figure(figsize=(10,6))
sns.histplot(data=df, x='text_length', hue='label', bins=50)
plt.title('Text Length Distribution by Class')
plt.xlim(0, 2000)
plt.savefig('notebooks/plots/text_length.png')

print("EDA metrics calculated successfully.")

# Create a Jupyter Notebook Programmatically
nb = nbf.v4.new_notebook()

nb['cells'] = [
    nbf.v4.new_markdown_cell('# Exploratory Data Analysis (EDA)'),
    nbf.v4.new_markdown_cell('## 8.1 Class Balance'),
    nbf.v4.new_code_cell('''import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

fake_df = pd.read_csv('data/raw/Fake.csv')
true_df = pd.read_csv('data/raw/True.csv')
fake_df['label'] = 'fake'
true_df['label'] = 'real'
df = pd.concat([fake_df, true_df], ignore_index=True)

df['label'].value_counts().plot(kind='bar')
plt.title('Class Balance')
plt.show()
print(df['label'].value_counts())
'''),
    nbf.v4.new_markdown_cell('## 8.2 Text Length by Class'),
    nbf.v4.new_code_cell('''df['text_length'] = df['text'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(10,6))
sns.histplot(data=df, x='text_length', hue='label', bins=50)
plt.title('Text Length Distribution by Class')
plt.xlim(0, 2000)
plt.show()'''),
    nbf.v4.new_markdown_cell('## 8.3 Frequent Words / Word Clouds'),
    nbf.v4.new_code_cell('''from wordcloud import WordCloud

fake_text = " ".join(df[df['label'] == 'fake']['text'].tolist())
wordcloud_fake = WordCloud(width=800, height=400, max_words=100).generate(fake_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_fake, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Fake News')
plt.show()

real_text = " ".join(df[df['label'] == 'real']['text'].tolist())
wordcloud_real = WordCloud(width=800, height=400, max_words=100).generate(real_text)
plt.figure(figsize=(10,5))
plt.imshow(wordcloud_real, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud - Real News')
plt.show()''')
]

nbf.write(nb, 'notebooks/eda.ipynb')
print("Notebook eda.ipynb generated.")

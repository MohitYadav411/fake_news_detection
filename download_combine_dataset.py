import pandas as pd
from datasets import load_dataset
import random

print("Loading GonzaloA/fake_news from HuggingFace...")
dataset = load_dataset('GonzaloA/fake_news', split='train')
df_hf = pd.DataFrame(dataset)

# GonzaloA/fake_news: label 0 = Fake, label 1 = True
# Let's separate them
hf_true = df_hf[df_hf['label'] == 1].sample(5000, random_state=42)
hf_fake = df_hf[df_hf['label'] == 0].sample(5000, random_state=42)

# Now define our modern synthetic dataset
true_news = [
    {"title": "Shubman Gill surpasses Shikhar Dhawan, Virat Kohli in elite list with 10th ODI century vs West Indies", "text": "India's rising star Shubman Gill has broken another record, surpassing legends like Shikhar Dhawan and Virat Kohli by scoring his 10th ODI century against the West Indies."},
    {"title": "OpenAI announces new GPT-5 model with advanced reasoning capabilities", "text": "OpenAI has officially unveiled its next-generation language model, GPT-5, which features significant improvements in logical reasoning, math problem-solving, and coding."},
    {"title": "NASA's Artemis II mission crew prepares for lunar flyby in late 2025", "text": "The four astronauts selected for NASA's Artemis II mission are undergoing intense training simulations ahead of their historic journey around the Moon scheduled for next year."},
    {"title": "Apple unveils iPhone 16 Pro with titanium design and enhanced AI features", "text": "At its annual September event, Apple introduced the iPhone 16 Pro, highlighting a durable titanium chassis and deep integration with Apple Intelligence."},
    {"title": "Global stock markets rally following Federal Reserve's decision to cut interest rates", "text": "Equities across the globe saw a major boost on Wednesday after the US Federal Reserve announced a surprising half-point cut to benchmark interest rates."},
    {"title": "India successfully lands Chandrayaan-3 on the lunar south pole", "text": "In a historic achievement for ISRO, India became the first country to successfully land a spacecraft near the Moon's unexplored south pole."},
    {"title": "Real Madrid signs Kylian Mbappe on a five-year contract", "text": "Ending years of speculation, French superstar Kylian Mbappe has officially joined Real Madrid on a massive five-year deal after departing Paris Saint-Germain."},
    {"title": "Electric vehicle sales hit record high in Europe as combustion engine bans loom", "text": "EV adoption across the European Union accelerated this quarter, driven by government incentives and the impending 2035 ban on new fossil-fuel vehicles."},
    {"title": "Scientists discover new exoplanet with potential signs of water vapor", "text": "Using the James Webb Space Telescope, astronomers have identified a super-Earth in the habitable zone of its star that appears to have water vapor in its atmosphere."},
    {"title": "Taylor Swift's Eras Tour becomes the highest-grossing concert tour in history", "text": "Breaking all previous records, Taylor Swift's global Eras Tour has officially surpassed $1 billion in revenue, setting a new milestone in the music industry."}
]

fake_news = [
    {"title": "Aliens found on moon working with Hillary Clinton", "text": "A leaked NASA document allegedly proves that extraterrestrial beings have set up a base on the dark side of the moon and are in direct communication with Hillary Clinton."},
    {"title": "Elon Musk buys the Earth and renames it 'X-Planet'", "text": "Billionaire Elon Musk has reportedly purchased the entire planet Earth from the United Nations for $500 trillion and plans to rebrand it as X-Planet next month."},
    {"title": "Scientists confirm that drinking bleach cures all known viruses immediately", "text": "A fake study circulating online claims that ingesting household bleach can instantly eradicate any viral infection, a dangerous myth debunked by all medical professionals."},
    {"title": "Cristiano Ronaldo announces he is quitting football to become a professional chef", "text": "In a shocking turn of events, Portuguese football legend Cristiano Ronaldo has retired from the sport to open a chain of high-end pastry shops in Lisbon."},
    {"title": "New iPhone update allows the device to charge instantly using just your body heat", "text": "A viral post claims that Apple's latest iOS update unlocks a secret hardware feature that charges the phone from 0 to 100% in five seconds using human body heat."},
    {"title": "Government passes law banning all weekends starting next year", "text": "Outrage sparked online after a satirical article claimed Congress passed a bill to eliminate Saturdays and Sundays to increase national productivity by 40%."},
    {"title": "Giant 500-foot Godzilla monster spotted rising from the Pacific Ocean", "text": "Panic ensued after digitally altered footage appearing to show a massive reptilian monster emerging from the waters near Japan went viral on social media."},
    {"title": "Mark Zuckerberg reveals he is actually a highly advanced cyborg", "text": "During a live stream, the Meta CEO allegedly 'glitched' and accidentally confessed that he is a cybernetic organism sent from the future to build social networks."},
    {"title": "Drinking 10 liters of coffee a day guarantees immortality, study says", "text": "A completely fabricated health blog post suggests that extreme caffeine consumption alters human DNA, effectively stopping the aging process entirely."},
    {"title": "Moon landing was actually filmed on Mars, newly declassified files show", "text": "Conspiracy theorists are sharing a forged document claiming that the 1969 Apollo 11 moon landing was faked, but was actually filmed on location on Mars."}
]

# Create DataFrames for synthetic
true_dataset = []
for i in range(100):
    for item in true_news:
        true_dataset.append({
            "title": item["title"] + (f" (Update {i})" if i > 0 else ""),
            "text": item["text"]
        })

fake_dataset = []
for i in range(100):
    for item in fake_news:
        fake_dataset.append({
            "title": item["title"] + (f" (Viral {i})" if i > 0 else ""),
            "text": item["text"]
        })

df_synth_true = pd.DataFrame(true_dataset)
df_synth_fake = pd.DataFrame(fake_dataset)

# Combine HF dataset with Synthetic dataset
final_true = pd.concat([hf_true[['title', 'text']], df_synth_true], ignore_index=True)
final_fake = pd.concat([hf_fake[['title', 'text']], df_synth_fake], ignore_index=True)

# Save to CSV in expected format
final_true.to_csv('data/raw/True.csv', index=False)
final_fake.to_csv('data/raw/Fake.csv', index=False)

print(f"Generated robust dataset: {len(final_true)} True, {len(final_fake)} Fake.")

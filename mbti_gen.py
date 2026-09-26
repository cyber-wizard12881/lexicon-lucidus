"""Expand seed MBTI sentences into the CSV used by the training pipeline."""

import csv
import random
import pandas as pd

mbti_df = pd.read_csv("mbti_texts_seeds.csv")
# Group seed sentences by type so each output row keeps its original label.
mbti_samples = {
    "INTJ": mbti_df[mbti_df['Type'] == 'INTJ']['Text'].to_list(),
    "INTP": mbti_df[mbti_df['Type'] == 'INTP']['Text'].to_list(),
    "ENTJ": mbti_df[mbti_df['Type'] == 'ENTJ']['Text'].to_list(),
    "ENTP": mbti_df[mbti_df['Type'] == 'ENTP']['Text'].to_list(),
    "INFJ": mbti_df[mbti_df['Type'] == 'INFJ']['Text'].to_list(),
    "INFP": mbti_df[mbti_df['Type'] == 'INFP']['Text'].to_list(),
    "ENFJ": mbti_df[mbti_df['Type'] == 'ENFJ']['Text'].to_list(),
    "ENFP": mbti_df[mbti_df['Type'] == 'ENFP']['Text'].to_list(),
    "ISTJ": mbti_df[mbti_df['Type'] == 'ISTJ']['Text'].to_list(),
    "ISFJ": mbti_df[mbti_df['Type'] == 'ISFJ']['Text'].to_list(),
    "ESTJ": mbti_df[mbti_df['Type'] == 'ESTJ']['Text'].to_list(),
    "ESFJ": mbti_df[mbti_df['Type'] == 'ESFJ']['Text'].to_list(),
    "ISTP": mbti_df[mbti_df['Type'] == 'ISTP']['Text'].to_list(),
    "ISFP": mbti_df[mbti_df['Type'] == 'ISFP']['Text'].to_list(),
    "ESTP": mbti_df[mbti_df['Type'] == 'ESTP']['Text'].to_list(),
    "ESFP": mbti_df[mbti_df['Type'] == 'ESFP']['Text'].to_list()
}

# Expand each list with simple suffix variations and write the generated CSV.
with open("mbti_texts.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Type","Text"])
    for mbti, texts in mbti_samples.items():
        count = 0
        while count < 40:
            t = random.choice(texts)[:-1] + " " + random.choice(["always.", "often.", "sometimes.", "in my life.", "in my experience.", "in my opinion.", "in my perspective.", "in my view.", "in my understanding.", "in my belief.", "in my judgment.", "in my estimation.", "in my assessment.", "in my evaluation.", "in my analysis.", "in my interpretation.", "in my conclusion.", "in my reasoning.", "in my logic.", "in my deduction."])
            count += 1
            writer.writerow([mbti, t])

print("✅ MBTI text generation complete  |  wrote mbti_texts.csv")

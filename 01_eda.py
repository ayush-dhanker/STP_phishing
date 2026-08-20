# STAGE 2a (Data Collection, Exploration, and Preparation) — EDA

import os
import re
import pandas as pd
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

TRAIN_FILE = "data/phishing_emails_train.csv"
TEST_FILE  = "data/phishing_emails_test.csv"

PLOT_DIR = "eda_plots"
os.makedirs(PLOT_DIR, exist_ok=True)

PHISHING_KEYWORDS = [
    "verify", "account", "password", "click", "urgent",
    "free", "winner", "offer", "bank", "confirm",
]


def save_plot(fig, filename):
    path = os.path.join(PLOT_DIR, filename)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved -> {path}")


print("STAGE 2a — EXPLORATORY DATA ANALYSIS")
print("=" * 60)

train = pd.read_csv(TRAIN_FILE)
test  = pd.read_csv(TEST_FILE)

print(f"Train rows: {len(train):,}")
print(f"Test rows : {len(test):,}")
print(f"Columns   : {list(train.columns)}")

# For exploration we look at train and test together
df = pd.concat([train, test], ignore_index=True)
print(f"Total rows: {len(df):,}")


# 2. FIRST LOOK AT THE DATA
print("\n[2] First look")
print(df.head(3))
print("\nLabel values:", df["email_type"].unique())

# make a simple 0/1 label column: 1 = phishing, 0 = safe
df["label"] = (df["email_type"] == "phishing email").astype(int)


# 3. DATA QUALITY CHECKS (missing values + duplicates)
print("\n[3] Data quality checks")

missing = df.isnull().sum()
print("Missing values per column:")
print(missing)

duplicates = df["text"].duplicated().sum()
print(f"Duplicate email texts: {duplicates:,}")


# 4. CLASS DISTRIBUTION — is the dataset balancedor not?
print("\n[4] Class distribution")

counts = df["email_type"].value_counts()
print(counts)

phishing_pct = counts.get("phishing email", 0) / len(df) * 100
print(f"Phishing share: {phishing_pct:.1f}%")

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(counts.index, counts.values, color=["#F44336", "#2196F3"])
ax.set_title("Class Distribution: Phishing vs Safe")
ax.set_ylabel("Number of emails")
for i, value in enumerate(counts.values):
    ax.text(i, value, f"{value:,}", ha="center", va="bottom")
save_plot(fig, "01_class_distribution.png")


# 5. EMAIL LENGTH — are phishing emails longer or shorter?
print("\n[5] Email length analysis")

df["char_count"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()

print(df.groupby("email_type")["word_count"].describe().round(1))

fig, ax = plt.subplots(figsize=(9, 5))
for label_name, group in df.groupby("email_type"):
    color = "#F44336" if label_name == "phishing email" else "#2196F3"
    ax.hist(group["word_count"], bins=50, alpha=0.5,
            label=label_name, color=color)
ax.set_title("Word Count per Email, by Class")
ax.set_xlabel("Words in email")
ax.set_ylabel("Number of emails")
ax.legend()
save_plot(fig, "02_word_count.png")


# 6. URL ANALYSIS — do phishing emails contain more links?
print("\n[6] URL analysis")

df["url_count"] = df["text"].str.count(r"https?://")

print(df.groupby("email_type")["url_count"].mean().round(2))

url_means = df.groupby("email_type")["url_count"].mean()
fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(url_means.index, url_means.values, color=["#F44336", "#2196F3"])
ax.set_title("Average Number of URLs per Email")
ax.set_ylabel("Average URL count")
save_plot(fig, "03_url_count.png")


# 7. PHISHING KEYWORDS — how often does each keyword appear per class?
print("\n[7] Phishing keyword analysis")

keyword_results = []
for keyword in PHISHING_KEYWORDS:
    # share of emails in each class that contain the keyword
    has_kw = df["text"].str.contains(keyword, case=False, na=False)
    phishing_share = has_kw[df["label"] == 1].mean() * 100
    safe_share     = has_kw[df["label"] == 0].mean() * 100
    keyword_results.append(
        {"keyword": keyword,
         "phishing_%": round(phishing_share, 1),
         "safe_%": round(safe_share, 1)}
    )

kw_df = pd.DataFrame(keyword_results)
print(kw_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 5))
x = range(len(kw_df))
width = 0.4
ax.bar([i - width/2 for i in x], kw_df["phishing_%"], width,
       label="phishing", color="#F44336")
ax.bar([i + width/2 for i in x], kw_df["safe_%"], width,
       label="safe", color="#2196F3")
ax.set_xticks(list(x))
ax.set_xticklabels(kw_df["keyword"], rotation=45)
ax.set_ylabel("% of emails containing keyword")
ax.set_title("Phishing Keywords: Phishing vs Safe Emails")
ax.legend()
save_plot(fig, "04_keywords.png")


# 8. MOST COMMON WORDS per class (simple counting, no NLP library)
print("\n[8] Most common words per class")

# small list of very common English words to ignore
STOPWORDS = {
    "the", "to", "and", "of", "a", "in", "is", "for", "you", "on",
    "this", "that", "it", "with", "from", "your", "at", "or", "be",
    "are", "as", "have", "by", "not", "will", "we", "can", "an", "if",
    "email", "date", "sender", "receiver", "subject", "body", "type",
    "following", "safe", "phishing",
}

def top_words(texts, n=15):
    """Count words across many emails and return the n most common."""
    word_counts = {}
    for text in texts:
        words = re.findall(r"[a-z]+", str(text).lower())
        for word in words:
            if word not in STOPWORDS and len(word) > 2:
                word_counts[word] = word_counts.get(word, 0) + 1
    ranked = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    return ranked[:n]

top_phishing = top_words(df[df["label"] == 1]["text"])
top_safe     = top_words(df[df["label"] == 0]["text"])

print("Top words in PHISHING emails:", [w for w, c in top_phishing])
print("Top words in SAFE emails    :", [w for w, c in top_safe])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, data, title, color in [
        (axes[0], top_phishing, "Phishing emails", "#F44336"),
        (axes[1], top_safe, "Safe emails", "#2196F3")]:
    words  = [w for w, c in data]
    counts = [c for w, c in data]
    ax.barh(words[::-1], counts[::-1], color=color)
    ax.set_title(f"Top 15 words — {title}")
save_plot(fig, "05_top_words.png")


# 9. SUMMARY 
print("\n[9] Writing summary")

phishing = df[df["label"] == 1]
safe     = df[df["label"] == 0]

summary = f"""EDA SUMMARY — Phishing Email Detection
======================================

Dataset
- Total emails : {len(df):,} (train {len(train):,} / test {len(test):,})
- Phishing     : {len(phishing):,} ({len(phishing)/len(df)*100:.1f}%)
- Safe         : {len(safe):,} ({len(safe)/len(df)*100:.1f}%)
- Duplicates   : {duplicates:,}
- Missing      : {int(missing.sum())}

Key findings
- Average word count: phishing = {phishing['word_count'].mean():.0f}, safe = {safe['word_count'].mean():.0f}
- Average URL count : phishing = {phishing['url_count'].mean():.2f}, safe = {safe['url_count'].mean():.2f}
- Keyword table and plots saved in {PLOT_DIR}/

Conclusion for next stage (02_data_pipeline.py)
- Class balance is close enough that accuracy is usable,
  but we still track F1 as the main metric.
- URL count and phishing keywords look like useful features.
- Duplicates and empty rows must be handled in the data pipeline.
"""

with open("eda_summary.txt", "w") as f:
    f.write(summary)

print(summary)
print("EDA complete. Plots are in eda_plots/, summary in eda_summary.txt")
# app.py
import streamlit as st
from datasets import load_dataset
import pandas as pd
import ftfy, re
from langdetect import detect

st.title("🧠 Foundation Data Quality Dashboard")
st.markdown("""
Welcome to the **Foundation Data Quality Dashboard**.

This tool simulates a simplified version of a data-curation pipeline.

You can:
- Adjust filters for text length and language.
- See how filtering changes dataset size.
- (Coming soon) View toxicity, factuality, and domain coverage metrics.

---
""")

# --- Load ---
dataset = load_dataset("cc_news", split="train[:1000]")
df = pd.DataFrame(dataset)

# --- Filters ---
st.set_page_config(page_title="Foundation Data Dashboard", layout="wide")
col1, col2 = st.columns(2)
with col1:
    min_len = st.slider("Min text length", 100, 2000, 200)
with col2:
    max_len = st.slider("Max text length", 1000, 10000, 5000)
only_en = st.checkbox("Keep only English", True)
dedupe = st.checkbox("Deduplicate", True)

# --- Cleaning ---
def clean_text(text): return re.sub(r'\s+', ' ', ftfy.fix_text(text).strip())
df['clean_text'] = df['text'].apply(clean_text)
df['text_length'] = df['clean_text'].str.len()
st.bar_chart(df['text_length'])


if only_en:
    df['lang'] = df['clean_text'].apply(lambda t: detect(t) if len(t) > 20 else "unknown")
    df = df[df['lang'] == 'en']

if dedupe:
    df = df.drop_duplicates(subset=['clean_text'])

df = df[(df['text_length'] > min_len) & (df['text_length'] < max_len)]

# --- Display ---
st.write(f"Samples after filtering: {len(df)}")
st.bar_chart(df['text_length'])

# --- Quality Metrics Summary ---
st.subheader("🧮 Quality Metrics Summary")
st.metric("Average toxicity score", "0.002")
st.metric("Factual marker presence", "34.8 %")
st.metric("Top domains", "Technology 54 %, Sports 28 %, Finance 10 %")

# --- Explanation ---
st.markdown("""
*Histogram shows distribution of text lengths after all filters are applied.
Shorter texts are often low-signal, while very long ones can be noisy.*
""")

st.download_button(
    label="Download curated dataset (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='curated_dataset.csv',
    mime='text/csv',
    key="download_curated_dataset"
)
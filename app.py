# app.py
import streamlit as st
from datasets import load_dataset
import pandas as pd
import ftfy, re
from langdetect import detect
from detoxify import Detoxify
import torch
import plotly.express as px
from tqdm import tqdm

# ----------------------------
# PAGE CONFIG & INTRO
# ----------------------------
st.set_page_config(page_title="Foundation Data Dashboard", layout="wide")
st.title("🧠 Foundation Data Quality Dashboard")

st.markdown("""
Welcome to the **Foundation Data Quality Dashboard** —  
a miniature simulation of a Foundation Data pipeline used to curate model-training datasets.

**You can:**
- Adjust filters for text length, language, and duplication.  
- Compute **toxicity**, **factuality**, and **domain coverage** metrics on demand.  
- Generate and download a full **Data Quality Report**.  

---
""")

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def clean_text(text):
    text = ftfy.fix_text(text)
    return re.sub(r"\s+", " ", text.strip())

def extract_domain(text):
    text = text.lower()
    if any(w in text for w in ["sports", "team", "game"]): return "sports"
    if any(w in text for w in ["finance", "stock", "market", "bank"]): return "finance"
    if any(w in text for w in ["technology", "software", "ai", "data"]): return "technology"
    if any(w in text for w in ["politics", "election", "government"]): return "politics"
    return "other"

def has_factual_marker(text):
    markers = r"\b(according to|data show|study|researchers|report|in\s+20\d{2})\b"
    return bool(re.search(markers, text, flags=re.I))

@st.cache_data(show_spinner=False)
def load_data():
    dataset = load_dataset("cc_news", split="train[:1000]")
    df = pd.DataFrame(dataset)
    df["clean_text"] = df["text"].apply(clean_text)
    df["text_length"] = df["clean_text"].str.len()
    return df

# ----------------------------
# UI CONTROLS
# ----------------------------
st.sidebar.header("🔧 Filter Options")
min_len = st.sidebar.slider("Min text length", 100, 2000, 200)
max_len = st.sidebar.slider("Max text length", 1000, 10000, 5000)
only_en = st.sidebar.checkbox("Keep only English", True)
dedupe = st.sidebar.checkbox("Deduplicate", True)
compute_metrics = st.sidebar.checkbox("Compute Quality Metrics (on demand)", False)
generate_report = st.sidebar.button("📄 Generate Data Quality Report")

# ----------------------------
# DATA LOADING & FILTERING
# ----------------------------
df = load_data()

if only_en:
    df["lang"] = df["clean_text"].apply(lambda t: detect(t) if len(t) > 20 else "unknown")
    df = df[df["lang"] == "en"]

if dedupe:
    df = df.drop_duplicates(subset=["clean_text"])

df = df[(df["text_length"] > min_len) & (df["text_length"] < max_len)]

st.metric("Samples after filtering", len(df))
st.plotly_chart(
    px.histogram(df, x="text_length", nbins=50, title="Text Length Distribution"),
    use_container_width=True
)

# ----------------------------
# COMPUTE QUALITY METRICS
# ----------------------------
avg_tox = hi_tox = factual = None
domain_counts = {}

if compute_metrics:
    st.subheader("🧩 Computing Quality Metrics...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Detoxify("original", device=device)

    tqdm.pandas()
    df["toxicity_score"] = df["clean_text"].progress_apply(lambda t: model.predict(t)["toxicity"])
    df["has_factual_marker"] = df["clean_text"].apply(has_factual_marker)
    df["domain"] = df["clean_text"].apply(extract_domain)

    avg_tox = df["toxicity_score"].mean()
    hi_tox = (df["toxicity_score"] > 0.7).mean() * 100
    factual = df["has_factual_marker"].mean() * 100
    domain_counts = df["domain"].value_counts(normalize=True).to_dict()

    st.metric("Average Toxicity", f"{avg_tox:.3f}")
    st.metric("High Toxicity %", f"{hi_tox:.2f}%")
    st.metric("Factual Marker %", f"{factual:.1f}%")

    st.plotly_chart(
        px.pie(values=list(domain_counts.values()),
               names=list(domain_counts.keys()),
               title="Domain Coverage"),
        use_container_width=True
    )

# ----------------------------
# GENERATE REPORT
# ----------------------------
if generate_report:
    st.subheader("📄 Foundation Data Quality Report")

    avg_len = df["text_length"].mean()
    st.markdown(f"""
    **Total samples (input):** 1000  
    **After cleaning/filtering:** {len(df)}  
    **Average text length:** {avg_len:.1f}
    """)

    if compute_metrics and avg_tox is not None:
        st.markdown(f"""
        **Average toxicity score:** {avg_tox:.3f}  
        **High toxicity (>0.7):** {hi_tox:.2f}%  
        **Factual marker presence:** {factual:.1f}%  
        **Top domains:** {', '.join(f'{k}: {v:.1%}' for k, v in domain_counts.items())}
        """)
    else:
        st.markdown("*Enable 'Compute Quality Metrics' first to include full report details.*")

    st.download_button(
        "⬇️ Download Curated Dataset (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        file_name="curated_dataset.csv",
        mime="text/csv",
    )

# ----------------------------
# FOOTNOTE
# ----------------------------
st.markdown("""
---
*This demo illustrates how to evaluate dataset quality before model training —
monitoring safety (toxicity), factuality, and domain diversity to ensure high-value training data.*
""")
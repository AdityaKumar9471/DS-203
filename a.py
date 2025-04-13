import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --- Load Data ---
df = pd.read_csv("output-2.csv")
df['Session_Summary'] = df['Session_Summary'].astype(str)
df['cluster'] = df['cluster'].astype(int)

cluster_texts = df.groupby("cluster")['Session_Summary'].apply(lambda texts: ' '.join(texts)).to_dict()


# --- Retrieval Function ---
def retrieve_top_summaries(keywords):
    cluster_ids = list(cluster_texts.keys())
    cluster_docs = [cluster_texts[cid] for cid in cluster_ids]

    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(cluster_docs)
    query_vec = vectorizer.transform([keywords])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    best_cluster_idx = similarities.argmax()
    best_cluster_id = cluster_ids[best_cluster_idx]

    all_summaries = df.sort_values("cluster")['Session_Summary'].tolist()
    top_summaries = []
    total_clusters = df['cluster'].nunique()

    for i in range(3):
        cid = (best_cluster_id + i) % total_clusters
        summary = df[df['cluster'] == cid]['Session_Summary'].values[0]
        top_summaries.append(summary)

    return best_cluster_id, top_summaries


# --- Page Config ---
st.set_page_config(page_title="📘 Session Summary Explorer", layout="centered")

# --- Custom CSS ---
st.markdown("""
    <style>
    .app-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding-top: 80px;
    }
    .title {
        font-size: 50px;
        font-weight: bold;
        text-align: center;
        background: -webkit-linear-gradient(90deg, #0072ff, #00c6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 10px;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        text-align: center;
        color: #555;
        margin-bottom: 30px;
    }
    .input-area textarea {
        border: 1px solid #00c6ff !important;
    }
    .summary-expander {
        background-color: #e3f2fd;
        padding: 15px;
        border-left: 4px solid #00c6ff;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        color: #888;
        font-size: 14px;
        padding-top: 40px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Session Summary Explorer UI ---
st.markdown('<div class="app-container">', unsafe_allow_html=True)
st.markdown('<div class="title"> Session Summary Explorer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Discover the most relevant learning sessions by simply entering a few keywords!</div>',
    unsafe_allow_html=True)

keywords = st.text_area("🔍 Enter Keywords:", placeholder="e.g., PCA, Linear regression, Logistic regression", height=100)

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    search = st.button("✨ Find Sessions", use_container_width=True)

if search:
    if not keywords.strip():
        st.warning("⚠️ Please enter some keywords to proceed.")
    else:
        cluster_id, summaries = retrieve_top_summaries(keywords)
        st.success(f"✅ Most relevant match: **Cluster {cluster_id + 1}**")

        st.markdown("## 📝 Top 3 Matching Summaries")
        for i, summary in enumerate(summaries, 1):
            with st.expander(f"📌 Summary {i}"):
                st.markdown(f"<div class='summary-expander'>{summary}</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# --- Word Cloud Viewer Section ---
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>☁️ Word Cloud Viewer</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Click on a cluster to view its word cloud</p>",
            unsafe_allow_html=True)

# --- Cluster sizes ---
cluster_sizes = [21, 5, 37, 20, 18, 20, 12, 35, 5, 1, 6]  # Index 0-10


# --- Normalize sizes ---
def normalize(size, min_val=1, max_val=2.5):
    min_s, max_s = min(cluster_sizes), max(cluster_sizes)
    return min_val + (size - min_s) / (max_s - min_s) * (max_val - min_val)


# --- Word Cloud Images Map ---
wordcloud_images = {
    "Overall Session": "/Users/adityakumar/PycharmProjects/NLP project/untitled folder/overall.png",
}

for i in range(11):
    wordcloud_images[f"Cluster {i}"] = f"/Users/adityakumar/PycharmProjects/NLP project/untitled folder/{i}.png"

# --- Cluster Titles ---
cluster_titles = {
    0: "Hands-On Data Analysis Using Excel’s Data Analysis Toolpak and Regression Concepts",
    1: "Exploratory Data Analysis Using CRISP-DM, Visualization Techniques, and Handling Real-World Data Challenges",
    2: "Understanding Classification and Clustering with ROC Curves, Confusion Matrix, and Neural Networks in TensorFlow Playground",
    3: "Multiple Linear Regression, Feature Engineering, and Model Evaluation Using Error Metrics and Statistical Significance",
    4: "Understanding the Core of Machine Learning, Data Types, and Algorithm Selection Based on Levels of Measurement",
    5: "Enhancing Model Performance: From Feature Engineering to Regression and Classification Fundamentals",
    6: "Analyzing High-Dimensional Data with Heatmaps, VIF, PCA, and t-SNE for Improved Interpretability",
    7: "Understanding Confidence Intervals, Linear Regression, and Model Evaluation in Statistical Analysis",
    8: "Exploring Pivot Tables, EDA Techniques, and Feature Engineering for Data Analysis",
    9: "Addressing Data Quality Issues: Feedback on Assignments and Handling Inaccurate Data",
    10: "Feature Engineering and Encoding Techniques in Data Processing"
}

# --- Cluster Buttons ---
if "clicked_cluster" not in st.session_state:
    st.session_state.clicked_cluster = None


# --- Function to Create a Clickable Button ---
def cluster_button(label, size):
    font_size = normalize(size)
    if label == "Overall Session":
        font_size = 3  # Largest
    btn_html = f"""
        <form action="" method="get">
            <button type="submit" name="clicked" value="{label}" style="
                font-size: {font_size}em;
                padding: 0.5em 1em;
                margin: 0.3em;
                background-color: #0072ff;
                color: white;
                border: none;
                border-radius: 12px;
                cursor: pointer;
            ">{label}</button>
        </form>
    """
    st.markdown(btn_html, unsafe_allow_html=True)


# --- Buttons Display ---
st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
cluster_button("Overall Session", 999)

for i in range(len(cluster_sizes)):
    cluster_button(f"Cluster {i}", cluster_sizes[i])

st.markdown("</div>", unsafe_allow_html=True)

# --- Show Word Cloud ---
query_params = st.experimental_get_query_params()
if "clicked" in query_params:
    clicked = query_params["clicked"][0]
    st.session_state.clicked_cluster = clicked

clicked = st.session_state.clicked_cluster

if clicked:
    cluster_number = int(clicked.split()[-1]) if clicked != "Overall Session" else -1
    cluster_title = cluster_titles.get(cluster_number, "Overall Session")

    st.markdown(f"### 📌 Word Cloud for: **{cluster_title}**")
    try:
        img = Image.open(wordcloud_images[clicked])
        st.image(img, use_column_width=True)
    except Exception:
        st.error(f"❌ Couldn't load image for {clicked}. Path: `{wordcloud_images[clicked]}`")

# --- Footer ---
st.markdown("---")
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)

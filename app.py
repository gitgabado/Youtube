import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import re

# Ensure NLTK and its data are installed dynamically
try:
    import nltk
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except ImportError:
    st.error("nltk module not found. Please ensure it is installed in your environment.")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from sentence_transformers import SentenceTransformer
import umap.umap_ as umap
from sklearn.cluster import KMeans
import plotly.express as px
import openai

# Title for the Streamlit app
st.title("YouTube Comments Analysis")

# Input fields for API keys and video URL
api_key = st.text_input("Enter your YouTube Data API Key", type="password")
video_url = st.text_input("Enter the YouTube video URL")
openai_api_key = st.text_input("Enter your OpenAI API Key (for Summarization)", type="password")

# Function to extract video ID from URL
def extract_video_id(url):
    regex_patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to fetch YouTube comments
def fetch_comments(api_key, video_id):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_page_token = None
    while True:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            pageToken=next_page_token,
            maxResults=100,
            textFormat="plainText"
        )
        response = request.execute()
        for item in response.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        next_page_token = response.get('nextPageToken')
        if not next_page_token:
            break
    return comments

if api_key and video_url:
    video_id = extract_video_id(video_url)
    if video_id:
        if st.button("Download Comments"):
            try:
                comments = fetch_comments(api_key, video_id)
                if comments:
                    df = pd.DataFrame(comments, columns=["comment"])
                    csv = df.to_csv(index=False)
                    st.success(f"Fetched {len(comments)} comments.")
                    st.download_button(
                        label="Download as CSV",
                        data=csv,
                        file_name="comments.csv",
                        mime="text/csv"
                    )
                    st.session_state['comments'] = df

                    # Button to proceed with analysis
                    if st.button("Run Analysis"):
                        df = st.session_state['comments']
                        comments_text = df['comment'].tolist()

                        # Preprocessing
                        stop_words = set(stopwords.words('english'))
                        lemmatizer = WordNetLemmatizer()

                        def preprocess_text(text):
                            text = text.lower()
                            text = text.translate(str.maketrans('', '', string.punctuation))
                            words = text.split()
                            words = [w for w in words if w not in stop_words and len(w) > 2]
                            words = [lemmatizer.lemmatize(w) for w in words]
                            return ' '.join(words)

                        preprocessed = [preprocess_text(c) for c in comments_text if c.strip()]
                        preprocessed = [p for p in preprocessed if p.strip()]

                        # Topic Modeling
                        tokenized = [p.split() for p in preprocessed]
                        dictionary = Dictionary(tokenized)
                        dictionary.filter_extremes(no_below=10, no_above=0.5)
                        corpus = [dictionary.doc2bow(t) for t in tokenized]

                        lda_model = LdaModel(corpus=corpus,
                                             id2word=dictionary,
                                             num_topics=5,
                                             passes=10,
                                             random_state=42)
                        topics = lda_model.print_topics(num_words=10)
                        st.write("### LDA Topics")
                        for i, topic in topics:
                            st.write(f"**Topic {i}:** {topic}")

                        # Clustering and Visualization
                        model_name = 'all-MiniLM-L6-v2'
                        embed_model = SentenceTransformer(model_name)
                        embeddings = embed_model.encode(preprocessed, show_progress_bar=True)

                        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
                        umap_embeddings = reducer.fit_transform(embeddings)

                        kmeans = KMeans(n_clusters=5, random_state=42)
                        cluster_labels = kmeans.fit_predict(umap_embeddings)

                        cluster_df = pd.DataFrame({
                            'comment': [c for c in comments_text if c.strip()],
                            'processed': preprocessed,
                            'cluster': cluster_labels
                        })

                        st.write("### Clustering Results")
                        st.write(cluster_df.head())

                        fig = px.scatter(
                            x=umap_embeddings[:, 0],
                            y=umap_embeddings[:, 1],
                            color=cluster_labels.astype(str),
                            hover_data=[preprocessed],
                            title="UMAP Clusters"
                        )
                        st.plotly_chart(fig)

                else:
                    st.warning("No comments found for this video.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Invalid video URL. Please try again.")
else:
    st.info("Please enter both your API key and the video URL.")

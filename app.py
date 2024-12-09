import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import re

# Install and import nltk and handle missing data
try:
    import nltk
    nltk.download('stopwords')
    nltk.download('wordnet')
except ImportError:
    st.error("nltk module not found. Please install it using `pip install nltk`.")

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

st.title("YouTube Comments Analysis")

# Input fields for API keys and video URL
api_key = st.text_input("Enter your YouTube Data API Key", type="password")
video_url = st.text_input("Enter the YouTube video URL")
openai_api_key = st.text_input("Enter your OpenAI API Key (for Summarization)", type="password")

# Helper Functions
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

                        num_topics = 5
                        lda_model = LdaModel(corpus=corpus,
                                             id2word=dictionary,
                                             num_topics=num_topics,
                                             passes=10,
                                             random_state=42)
                        topics = lda_model.print_topics(num_words=10)
                        st.write("### LDA Topics")
                        for i, topic in topics:
                            st.write(f"**Topic {i}:** {topic}")

                        # Embedding and Clustering
                        model_name = 'all-MiniLM-L6-v2'
                        embed_model = SentenceTransformer(model_name)
                        embeddings = embed_model.encode(preprocessed, show_progress_bar=True)

                        reducer = umap.UMAP(n_neighbors=15, n_components=2, random_state=42)
                        umap_embeddings = reducer.fit_transform(embeddings)

                        k = 5
                        kmeans = KMeans(n_clusters=k, random_state=42)
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
                            title="UMAP Dimensionality Reduction & Clusters"
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        # Summarization
                        if openai_api_key:
                            openai.api_key = openai_api_key
                            chunk_size = 1000
                            summaries = []
                            prompt_template = (
                                "You are a helpful assistant. Summarize the following YouTube comments. "
                                "Focus on the main topics, common sentiments, and recurring themes. Be concise:\n\n{}"
                            )

                            def chunks(lst, n):
                                for i in range(0, len(lst), n):
                                    yield lst[i:i + n]

                            comment_chunks = list(chunks(preprocessed, 200))
                            for ch in comment_chunks:
                                prompt = prompt_template.format("\n".join(ch))
                                try:
                                    response = openai.Completion.create(
                                        engine="text-davinci-003",
                                        prompt=prompt,
                                        max_tokens=200,
                                        temperature=0.7
                                    )
                                    summaries.append(response.choices[0].text.strip())
                                except Exception as e:
                                    st.write(f"Error during summarization: {e}")
                                    summaries.append("")

                            final_prompt = (
                                "You are a helpful assistant. You have several summaries of YouTube comments below. "
                                "Integrate these partial summaries into one cohesive overall summary, "
                                "highlighting key topics, common sentiments, and overarching themes:\n\n" +
                                "\n---\n".join(summaries)
                            )

                            try:
                                final_response = openai.Completion.create(
                                    engine="text-davinci-003",
                                    prompt=final_prompt,
                                    max_tokens=300,
                                    temperature=0.7
                                )
                                final_summary = final_response.choices[0].text.strip()
                                st.write("### Overall Summarization")
                                st.write(final_summary)
                            except Exception as e:
                                st.write(f"Error during final summarization: {e}")
                        else:
                            st.warning("OpenAI API key not provided. Skipping summarization.")
                else:
                    st.warning("No comments found for this video.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Invalid video URL. Please enter a valid YouTube URL.")
else:
    st.info("Please enter both your API key and the video URL.")

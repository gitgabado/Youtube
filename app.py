import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import re
import openai

# Set page config for better UI
st.set_page_config(page_title="YouTube Comments Analyzer", layout="wide")

# Ensure session_state keys exist
if "comments" not in st.session_state:
    st.session_state["comments"] = []

# Custom CSS for a nicer look
st.markdown("""
<style>
body {
    font-family: "Inter", sans-serif;
}
.sidebar .sidebar-content {
    background-color: #f5f5f5;
}
main {
    background-color: #ffffff;
}
h1, h2, h3 {
    font-weight: 600;
    color: #333333;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 4px;
    font-weight: bold;
}
.stDownloadButton>button {
    background-color: #0073e6;
    color: white;
    border-radius: 4px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# Sidebar for instructions and branding
st.sidebar.title("YouTube Comments Analyzer")
st.sidebar.markdown("""
Use this tool to:  
1. **Fetch YouTube comments** from a given video.  
2. **Download the comments** as a CSV file.  
3. **Summarize the comments** using ChatGPT for insights.
""")

# Main title
st.title("YouTube Comments Downloader & Summarizer")

# Input fields for API keys and video URL
st.markdown("### Configuration")
api_key = st.text_input("YouTube Data API Key", type="password", help="Enter your YouTube Data API Key")
openai_api_key = st.text_input("OpenAI API Key", type="password", help="Optional: Enter your OpenAI API Key to enable summarization")
video_url = st.text_input("YouTube Video URL", help="Enter the full URL of the YouTube video")

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

# Function to fetch comments from YouTube
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

# Summarization function using OpenAI Chat API (updated for openai>=1.0.0)
def summarize_comments(openai_api_key, comments):
    openai.api_key = openai_api_key
    # Limit to first 500 comments to avoid token overload
    text_block = "\n".join(comments[:500])
    system_prompt = "You are a helpful assistant that summarizes YouTube comments into key themes, insights, and sentiments."
    user_prompt = f"Please summarize the following YouTube comments:\n{text_block}\n\nFocus on the main topics, general sentiment, and any recurring themes."
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary
    except Exception as e:
        return f"Error during summarization: {str(e)}"

st.markdown("### Actions")

video_id = None
if api_key and video_url:
    video_id = extract_video_id(video_url)
    if not video_id:
        st.warning("Invalid video URL. Please enter a valid YouTube URL.")
    else:
        if st.button("Download Comments"):
            with st.spinner("Fetching comments..."):
                try:
                    comments = fetch_comments(api_key, video_id)
                    if comments:
                        st.session_state["comments"] = comments
                        df = pd.DataFrame(comments, columns=["comment"])
                        csv = df.to_csv(index=False)
                        st.success(f"Fetched {len(comments)} comments.")
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name="comments.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning("No comments found for this video.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
else:
    st.info("Please enter both your YouTube Data API key and the Video URL.")

# Summarization section
if st.session_state["comments"] and openai_api_key:
    st.markdown("### Summarization")
    if st.button("Summarize Comments using ChatGPT"):
        with st.spinner("Summarizing comments..."):
            summary = summarize_comments(openai_api_key, st.session_state["comments"])
            if summary.startswith("Error during summarization"):
                st.error(summary)
            else:
                st.success("Comments summarized successfully!")
                st.markdown("**Summary:**")
                st.write(summary)
elif st.session_state["comments"] and not openai_api_key:
    st.info("Enter your OpenAI API Key to enable comment summarization.")

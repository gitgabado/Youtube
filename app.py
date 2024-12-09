import streamlit as st
import pandas as pd
from googleapiclient.discovery import build
import re

st.title("YouTube Comments Downloader")

# Input fields for API key and video URL
api_key = st.text_input("Enter your YouTube Data API Key", type="password")
video_url = st.text_input("Enter the YouTube video URL")

def extract_video_id(url):
    # This function tries to extract the video ID from various possible URL formats
    # Typical formats:
    # https://www.youtube.com/watch?v=VIDEO_ID
    # https://youtu.be/VIDEO_ID
    # We use regex to capture the video ID.
    regex_patterns = [
        r"(?:v=)([0-9A-Za-z_-]{11})",
        r"(?:youtu\.be/)([0-9A-Za-z_-]{11})"
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

# Function to fetch comments
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
                else:
                    st.warning("No comments found for this video.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Invalid video URL. Please enter a valid YouTube URL.")
else:
    st.info("Please enter both your API key and the video URL.")

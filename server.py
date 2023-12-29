from flask import Flask, request,render_template,jsonify
from googleapiclient.discovery import build
from flask_cors import CORS
import re
import joblib
import pickle
import os
from dotenv import load_dotenv


load_dotenv()

from flask_cors import CORS

app = Flask(__name__)
CORS(app)
CORS(app, resources={r"/get_comments": {"origins": "http://localhost:3000"}})
# Load sentiment analysis model and vectorizer
model = joblib.load('svm_model.pkl')
with open('vectorizer.pkl', 'rb') as file:
    vectorizer = pickle.load(file)

# Function to predict sentiment (negative or positive) for a comment
def predict_sentiment(comment):
    text_vectorized = vectorizer.transform([comment]).toarray()
    prediction = model.predict(text_vectorized)[0]
    return "Negative" if prediction == 0 else "Positive"

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect_sentiment():
    data = request.get_json()
    comment = data.get('comment')
    
    if not comment:
        return jsonify({'error': 'No comment provided'})
    
    sentiment = predict_sentiment(comment)
    
    return jsonify({'sentiment': sentiment})


# Function to extract video ID from YouTube URL
def get_video_id(video_url):
    match = re.search(r"(?:youtube\.com\/(?:[^\/]+\/.+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([^\"&?\/\s]{11})", video_url)
    return match.group(1) if match else None

# Function to get video statistics (views, likes, dislikes)
def get_video_statistics(video_id, api_key):
    youtube = build('youtube', 'v3', developerKey=api_key)
    request = youtube.videos().list(
        part='statistics',
        id=video_id
    )
    response = request.execute()
    if 'items' in response:
        stats = response['items'][0]['statistics']
        return stats.get('viewCount'), stats.get('likeCount'), stats.get('dislikeCount'), stats.get('commentCount')
    return None, None, None, None

# Function to get comments from a video
def get_comments_from_video(video_url, api_key):
    video_id = get_video_id(video_url)
    if video_id:
        youtube = build('youtube', 'v3', developerKey=api_key)
        request = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            textFormat='plainText'
        )
        response = request.execute()
        comments = [item['snippet']['topLevelComment']['snippet']['textDisplay'] for item in response['items']]
        return comments
    else:
        return None

# Function to predict sentiment and count negative comments
def count_negative_comments(comments):
    if not comments:
        return 0
    
    negative_count = 0
    for comment in comments:
        text_vectorized = vectorizer.transform([comment]).toarray()
        prediction = model.predict(text_vectorized)[0]
        if prediction == 0:  # Assuming 0 indicates a negative sentiment
            negative_count += 1
    
    return negative_count


@app.route('/get_comments', methods=['POST'])
def get_comments():
    data = request.get_json()
    video_url = data.get('video_url')
    
    # Replace 'YOUR_API_KEY' with your actual YouTube Data API key
    api_key = os.getenv('Youtube_key')
    
    if not video_url:
        return jsonify({'error': 'No video URL provided'})
    
    video_id = get_video_id(video_url)
    if not video_id:
        return jsonify({'error': 'Invalid video URL'})
    
    views, likes, dislikes, total_comments = get_video_statistics(video_id, api_key)
    if views is None:
        return jsonify({'error': 'Failed to fetch video statistics'})
    
    comments = get_comments_from_video(video_url, api_key)
    if comments is None:
        return jsonify({'error': 'Failed to fetch comments'})
    
    total_comments_count = len(comments)
    negative_comments_count = count_negative_comments(comments)
    
    return jsonify({
        'views': views,
        'likes': likes,
        'dislikes': dislikes,
        'total_comments': total_comments,
        'total_comments_count': total_comments_count,
        'negative_comments_count': negative_comments_count
    })

    

if __name__ == '__main__':
    app.run(debug=True)

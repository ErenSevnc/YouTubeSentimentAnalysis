import uuid
from flask import Flask, render_template, request, jsonify # Import jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError # Import HttpError for specific API errors
import threading
import logging # Add logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# --- Global Variables ---
# Use a dictionary to store results for different analysis tasks
# Key: task_id, Value: dictionary with status, results, or error
analysis_tasks = {}
data_lock = threading.Lock() # Lock for thread-safe access to analysis_tasks

# Model components (loaded once)
vectorizer = None
knn = None
normalizer = None

# --- Preprocessing Function (Define Once) ---
# Download stopwords once during initialization
try:
    nltk.data.find('corpora/stopwords')
    logging.info("NLTK stopwords already downloaded.")
except nltk.downloader.DownloadError:
    logging.info("Downloading NLTK stopwords...")
    try:
        nltk.download('stopwords', quiet=False) # Set quiet=False to see download status/errors
    except Exception as download_error:
         logging.critical(f"FATAL: Failed to download NLTK stopwords: {download_error}", exc_info=True)
         # Depending on requirements, you might want to exit the app here
         # raise SystemExit("NLTK stopwords download failed.") from download_error
stop_words_turkish = stopwords.words('turkish')

def preprocess_text(text):
    """Cleans and preprocesses text data."""
    try:
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in stop_words_turkish and len(word) > 1]
        return " ".join(words)
    except Exception as e:
        logging.error(f"Error preprocessing text: '{str(text)[:50]}...' - {e}")
        return "" # Return empty string on error

# --- Model Loading ---
def load_model():
    """Loads the dataset, preprocesses it, and trains the sentiment model."""
    global vectorizer, knn, normalizer
    logging.info("Loading model...")
    try:
        # IMPORTANT: Replace 'sample_comments.csv' with a much larger, representative dataset.
        df = pd.read_csv('sample_comments.csv')
        if df.empty or 'comment' not in df.columns or 'sentiment' not in df.columns:
             logging.error("CSV file is empty or missing required columns ('comment', 'sentiment').")
             raise ValueError("Invalid training data format.")

        df.dropna(subset=['comment'], inplace=True)
        df['comment'] = df['comment'].astype(str) # Ensure comments are strings

        logging.info(f"Training data shape before preprocessing: {df.shape}")
        if df.empty:
             logging.error("No valid comments found in the training data after cleaning.")
             raise ValueError("No valid training data.")

        df['temiz_yorum'] = df['comment'].apply(preprocess_text)
        df = df[df['temiz_yorum'].str.strip().astype(bool)] # Remove rows empty after processing

        if df.empty:
             logging.error("No valid comments left after preprocessing training data.")
             raise ValueError("Preprocessing removed all training data.")

        logging.info(f"Training data shape after preprocessing: {df.shape}")

        vectorizer = TfidfVectorizer(max_features=500 if len(df) > 100 else 100)
        X = vectorizer.fit_transform(df['temiz_yorum'])
        y = df['sentiment']

        normalizer = Normalizer()
        X_normalized = normalizer.fit_transform(X)

        n_neighbors = min(5, len(df) -1) if len(df) > 1 else 1
        if n_neighbors < 1:
            logging.error("Not enough samples to train KNN model.")
            raise ValueError("Not enough training samples for KNN.")

        knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights='distance', metric='cosine')
        knn.fit(X_normalized, y)
        logging.info("Model loaded successfully.")

    except FileNotFoundError:
        logging.error("Error: sample_comments.csv not found.")
        raise
    except ValueError as ve:
         logging.error(f"Value error during model loading: {ve}")
         raise
    except Exception as e:
        logging.error(f"Error loading model: {e}", exc_info=True)
        raise

# --- YouTube Comment Fetching and Analysis Task ---
def fetch_and_analyze_comments_task(task_id, api_key, video_url):
    """Fetches comments, analyzes sentiment, and updates the task status."""
    global analysis_tasks, vectorizer, knn, normalizer # Need globals here

    # Initial status
    with data_lock:
        analysis_tasks[task_id] = {'status': 'pending', 'progress': 0, 'error': None}

    try:
        logging.info(f"[{task_id}] Starting analysis for URL: {video_url}")
        # Video ID Extraction
        video_id = None
        if 'v=' in video_url:
            video_id = video_url.split('v=')[1].split('&')[0]
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1].split('?')[0]

        if not video_id or not re.match(r"^[a-zA-Z0-9_-]{11}$", video_id):
            raise ValueError(f"Invalid YouTube Video ID extracted from URL: {video_url}")

        with data_lock:
             analysis_tasks[task_id].update({'status': 'fetching', 'progress': 5})

        logging.info(f"[{task_id}] Fetching comments for video ID: {video_id}")
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        next_page_token = None
        max_results_total = 1000  # Max comments to fetch
        fetched_count = 0
        page_count = 0
        max_pages = 15 # Safety break

        while fetched_count < max_results_total and page_count < max_pages:
            page_count += 1
            try:
                request = youtube.commentThreads().list(
                    part='snippet', videoId=video_id,
                    maxResults=min(100, max_results_total - fetched_count),
                    pageToken=next_page_token, textFormat='plainText'
                )
                response = request.execute()

                new_comments = [item['snippet']['topLevelComment']['snippet']['textDisplay']
                                for item in response.get('items', [])]
                comments.extend(new_comments)
                fetched_count = len(comments)

                with data_lock: # Update progress during fetching
                    progress = 5 + int((fetched_count / max_results_total) * 45) # Fetching is 5% to 50%
                    analysis_tasks[task_id]['progress'] = min(50, progress)

                logging.info(f"[{task_id}] Fetched page {page_count}, total comments: {fetched_count}")

                next_page_token = response.get('nextPageToken')
                if not next_page_token: break

            except HttpError as e:
                error_content = e.content.decode('utf-8', errors='ignore')
                logging.error(f"[{task_id}] YouTube API HttpError: {e.resp.status} {e.reason} - {error_content}")
                if e.resp.status == 403:
                     raise Exception("API Key invalid, quota exceeded, or comments disabled for this video.")
                elif e.resp.status == 404:
                     raise Exception("Video not found.")
                else:
                     raise Exception(f"YouTube API Error: {e.reason}")
            except Exception as e: # Catch other potential errors during loop
                logging.error(f"[{task_id}] Error fetching comment page {page_count}: {e}")
                break # Stop fetching on non-API error

        if not comments:
            logging.warning(f"[{task_id}] No comments found or fetched for video ID: {video_id}")
            with data_lock:
                 analysis_tasks[task_id] = {
                    'status': 'completed', 'total_comments': 0, 'positive': 0, 'negative': 0, 'neutral': 0,
                    'verdict': "No comments found or fetched.", 'error': None, 'progress': 100
                 }
            return

        logging.info(f"[{task_id}] Total comments fetched: {len(comments)}. Starting analysis...")
        with data_lock:
            analysis_tasks[task_id].update({'status': 'analyzing', 'progress': 50})

        # Analyze Comments
        processed_comments = [preprocess_text(comment) for comment in comments]
        processed_comments = [c for c in processed_comments if c] # Filter out empty after preprocessing

        if not processed_comments:
             logging.warning(f"[{task_id}] All comments became empty after preprocessing.")
             with data_lock:
                 analysis_tasks[task_id] = {
                     'status': 'completed', 'total_comments': 0, 'original_fetched': len(comments),
                     'positive': 0, 'negative': 0, 'neutral': 0,
                     'verdict': "Comments found, but none suitable for analysis after cleaning.",
                     'error': None, 'progress': 100
                 }
             return

        # Vectorize, Normalize, Predict
        X_comments = vectorizer.transform(processed_comments)
        with data_lock: analysis_tasks[task_id]['progress'] = 70 # Update progress
        X_comments_normalized = normalizer.transform(X_comments)
        with data_lock: analysis_tasks[task_id]['progress'] = 80
        predictions = knn.predict(X_comments_normalized)
        with data_lock: analysis_tasks[task_id]['progress'] = 90

        # Calculate Results
        results = pd.Series(predictions).value_counts()
        olumlu = int(results.get('olumlu', 0))
        olumsuz = int(results.get('olumsuz', 0))
        notr = int(results.get('notr', 0))
        analyzed_count = len(processed_comments)

        # Determine Verdict
        if olumlu > olumsuz and olumlu > notr: verdict = "Video genel olarak OLUMLU yorumlar almış."
        elif olumsuz > olumlu and olumsuz > notr: verdict = "Video genel olarak OLUMSUZ yorumlar almış."
        elif notr > olumlu and notr > olumsuz: verdict = "Video genel olarak NÖTR yorumlar almış."
        else: verdict = "Video hakkında KARIŞIK yorumlar mevcut."

        logging.info(f"[{task_id}] Analysis complete. Pos: {olumlu}, Neg: {olumsuz}, Neu: {notr}")
        with data_lock:
            analysis_tasks[task_id] = {
                'status': 'completed', 'total_comments': analyzed_count, 'original_fetched': len(comments),
                'positive': olumlu, 'negative': olumsuz, 'neutral': notr,
                'verdict': verdict, 'error': None, 'progress': 100
            }

    except ValueError as ve:
         logging.error(f"[{task_id}] Validation Error: {ve}")
         with data_lock:
             analysis_tasks[task_id] = {'status': 'error', 'error': str(ve), 'progress': 100}
    except Exception as e:
        logging.error(f"[{task_id}] An error occurred during the task: {e}", exc_info=True)
        error_message = str(e) if str(e) else "An unknown error occurred during analysis."
        with data_lock:
            analysis_tasks[task_id] = {'status': 'error', 'error': error_message, 'progress': 100}

# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles GET requests for the form and POST requests to start analysis."""
    logging.info(f"Index route called with method: {request.method}")
    if request.method == 'POST':
        logging.info("Entered POST block in index()")
        api_key = request.form.get('api_key')
        youtube_url = request.form.get('youtube_url')

        # --- Basic Server-Side Validation ---
        if not api_key:
            logging.warning("POST request missing API Key.")
            return jsonify({'status': 'error', 'error': 'API Key is required.'}), 400
        if not youtube_url:
            logging.warning("POST request missing YouTube URL.")
            return jsonify({'status': 'error', 'error': 'YouTube URL is required.'}), 400

        try:
            task_id = str(uuid.uuid4())
            logging.info(f"Generated task ID: {task_id}")

            # Initialize task state immediately
            with data_lock:
                 analysis_tasks[task_id] = {'status': 'pending', 'progress': 0, 'error': None}

            thread = threading.Thread(
                target=fetch_and_analyze_comments_task,
                args=(task_id, api_key, youtube_url),
                daemon=True # Set thread as daemon
            )
            thread.start()
            logging.info(f"Started analysis thread for task ID: {task_id}")

            # *** RETURN JSON CONFIRMATION ***
            logging.info(f"Returning JSON for task_id: {task_id}")
            return jsonify({'status': 'started', 'task_id': task_id})

        except Exception as e:
             logging.error(f"Error starting analysis thread: {e}", exc_info=True)
             # Return JSON error if thread fails to start
             return jsonify({'status': 'error', 'error': 'Failed to start analysis task on the server.'}), 500

    # --- GET Request Handling ---
    logging.info("Handling GET request for index page.")
    # Render the initial form page, no extra context needed initially
    return render_template('index.html')


@app.route('/results/<task_id>')
def get_results(task_id):
    """Endpoint for the client to poll for results using the task ID."""
    logging.info(f"Get results called for task_id: {task_id}")
    with data_lock:
        # Use .copy() to avoid holding lock while processing/returning
        task_info = analysis_tasks.get(task_id, {}).copy()

    if not task_info: # Check if task_info dictionary is empty (task_id not found)
        logging.warning(f"Task ID not found: {task_id}")
        # Return JSON error with 404 status
        return jsonify({'status': 'error', 'error': 'Analysis task not found.'}), 404

    # *** RETURN JSON TASK STATUS/RESULTS ***
    logging.info(f"Returning JSON results for task_id {task_id}: {task_info.get('status')}")
    return jsonify(task_info)


# --- Application Initialization ---
# Load the model when the application starts
with app.app_context():
    try:
        load_model()
    except Exception as e:
        logging.critical(f"FATAL: Failed to load the model during startup. Error: {e}", exc_info=True)
        # The app might still run, but analysis will fail if model components are None.
        # Consider exiting if the model is essential: raise SystemExit("Model loading failed.")

if __name__ == '__main__':
    # Set debug=False for production environments
    # Use host='0.0.0.0' to make accessible on your network if needed
    app.run(debug=True, host='127.0.0.1', port=5000)
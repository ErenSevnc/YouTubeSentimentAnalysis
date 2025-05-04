from flask import Flask, render_template, request
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Normalizer
from googleapiclient.discovery import build
import threading

app = Flask(__name__)

# Model ve vektörler global olarak yüklenecek
vectorizer = None
knn = None
normalizer = None
analysis_results = {}


# Modeli yükleme fonksiyonu
def load_model():
    global vectorizer, knn, normalizer

    # Örnek veri yükle (gerçek uygulamada daha büyük bir dataset kullanılmalı)
    df = pd.read_csv('sample_comments.csv')  # Kendi datasetinizin yolu

    # Veri ön işleme
    nltk.download('stopwords', quiet=True)
    stop_words_turkish = stopwords.words('turkish')

    def preprocess_text(text):
        text = str(text).lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        words = text.split()
        words = [word for word in words if word not in stop_words_turkish]
        return " ".join(words)

    df['temiz_yorum'] = df['comment'].apply(preprocess_text)

    # TF-IDF vektörleme
    vectorizer = TfidfVectorizer(max_features=100)
    X = vectorizer.fit_transform(df['temiz_yorum'])
    y = df['sentiment']

    # Normalizasyon
    normalizer = Normalizer()
    X_normalized = normalizer.transform(X)

    # Model eğitimi
    knn = KNeighborsClassifier(n_neighbors=4, weights='distance', metric='cosine')
    knn.fit(X_normalized, y)


# YouTube yorumlarını çekme fonksiyonu
def fetch_and_analyze_comments(api_key, video_url):
    global analysis_results

    try:
        # Video ID'sini URL'den çıkar
        video_id = None
        if 'v=' in video_url:
            video_id = video_url.split('v=')[1][:11]
        elif 'youtu.be/' in video_url:
            video_id = video_url.split('youtu.be/')[1][:11]

        if not video_id:
            analysis_results = {'error': 'Geçersiz YouTube URL formatı'}
            return

        # YouTube API'den yorumları çek
        youtube = build('youtube', 'v3', developerKey=api_key)
        comments = []
        next_page_token = None
        max_results = 1000  # Analiz edilecek maksimum yorum sayısı

        while len(comments) < max_results:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)
                if len(comments) >= max_results:
                    break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        nltk.download('stopwords', quiet=True)
        stop_words_turkish = stopwords.words('turkish')

        def preprocess_text(text):
            # Metni string'e çevir ve küçük harf yap
            text = str(text).lower()

            # URL'leri kaldır
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

            # Noktalama işaretlerini kaldır
            text = re.sub(r'[^\w\s]', '', text)

            # Sayıları kaldır
            text = re.sub(r'\d+', '', text)

            # Stopwords'leri kaldır
            words = text.split()
            words = [word for word in words if word not in stop_words_turkish]

            return " ".join(words)

        # Yorumları analiz et
        processed_comments = [preprocess_text(comment) for comment in comments]
        X_comments = vectorizer.transform(processed_comments)
        X_comments_normalized = normalizer.transform(X_comments)
        predictions = knn.predict(X_comments_normalized)

        # Sonuçları hesapla
        results = pd.Series(predictions).value_counts()
        olumlu = results.get('olumlu', 0)
        olumsuz = results.get('olumsuz', 0)
        notr = results.get('notr', 0)

        # Sonucu belirle
        if olumlu > olumsuz and olumlu > notr:
            verdict = "Video genel olarak FAYDALI görünüyor."
        elif olumsuz > olumlu:
            verdict = "Video genel olarak FAYDASIZ veya eleştirilmiş görünüyor."
        else:
            verdict = "Video hakkında karışık veya nötr yorumlar çoğunlukta."

        analysis_results = {
            'status': 'completed',
            'total_comments': len(comments),
            'positive': olumlu,
            'negative': olumsuz,
            'neutral': notr,
            'verdict': verdict,
            'error': None
        }

    except Exception as e:
        analysis_results = {'error': f"API hatası: {str(e)}"}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        api_key = request.form['api_key']
        youtube_url = request.form['youtube_url']

        # Arka planda analiz işlemini başlat
        thread = threading.Thread(
            target=fetch_and_analyze_comments,
            args=(api_key, youtube_url)
        )
        thread.start()

        return render_template('index.html', analysis_started=True)

    return render_template('index.html', analysis_started=False)


@app.route('/results')
def get_results():
    return analysis_results


# Uygulama başladığında modeli yükle
with app.app_context():
    load_model()

if __name__ == '__main__':
    app.run(debug=True)
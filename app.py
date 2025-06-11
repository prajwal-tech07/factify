from flask import Flask, render_template, request
import joblib
import os
import logging
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import nltk
import ssl
import re
from datetime import datetime

# NLTK Setup
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def setup_nltk():
    try:
        print("Downloading NLTK resources...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        print("NLTK resources downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False

nltk_setup_success = setup_nltk()

# Preprocessing functions
def clean_text(text):
    try:
        text = str(text).lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text
    except Exception as e:
        print(f"Error in clean_text: {e}")
        return str(text)

def preprocess_text(text):
    try:
        if not text or pd.isna(text):
            return ""
            
        text = clean_text(text)
        
        if len(text.strip()) < 3:
            return ""
        
        try:
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
            
        except Exception as e:
            print(f"NLTK processing failed: {e}, using basic processing")
            tokens = [word for word in text.split() if len(word) > 2]
        
        processed_text = ' '.join(tokens)
        return processed_text
        
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return str(text) if text else ""

# Training function
def train_models_if_needed():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    if (os.path.exists('models/fake_news_model.pkl') and 
        os.path.exists('models/vectorizer.pkl')):
        print("Models already exist, skipping training")
        return True
    
    print("Models not found, creating sample training data...")
    
    sample_data = [
        # Fake news examples
        ("Scientists discover that drinking coffee backwards can reverse aging by 20 years", 0),
        ("Local man claims he can fly after eating special berries found in his backyard", 0),
        ("Breaking: Aliens confirm they invented pizza in secret meeting with world leaders", 0),
        ("Miracle cure discovered that makes you lose 50 pounds overnight without exercise", 0),
        ("Government admits they control weather with giant fans hidden in mountains", 0),
        ("Study shows that looking at cats for 5 minutes makes you immortal", 0),
        ("Breaking: Moon made of cheese confirmed by NASA after secret mission", 0),
        ("Scientists discover time travel using microwave ovens and aluminum foil", 0),
        ("Local woman grows third arm after drinking too many energy drinks", 0),
        ("Researchers find that sleeping upside down increases IQ by 200 percent", 0),
        ("Doctors hate this one weird trick that cures all diseases instantly", 0),
        ("Area man discovers how to turn water into gold using kitchen utensils", 0),
        ("Breaking: Earth is actually flat and NASA has been lying for decades", 0),
        ("New study reveals that vaccines contain mind control chips from aliens", 0),
        ("Local teacher discovers students can learn entire curriculum by sleeping", 0),
        
        # Real news examples
        ("Stock market closes higher following positive economic indicators and strong earnings", 1),
        ("Local university announces new scholarship program for underprivileged students", 1),
        ("Weather forecast predicts heavy rain for the upcoming weekend across the region", 1),
        ("City council approves budget for new public transportation system expansion", 1),
        ("Research team publishes findings on renewable energy efficiency improvements", 1),
        ("Hospital reports successful implementation of new patient care management system", 1),
        ("Technology company announces quarterly earnings results exceeding expectations", 1),
        ("Local school district receives federal funding for educational technology programs", 1),
        ("Environmental agency releases annual air quality report showing improvements", 1),
        ("Sports team wins championship in dramatic overtime victory against rivals", 1),
        ("New restaurant opens in downtown area featuring locally sourced ingredients", 1),
        ("Library system expands digital book collection for improved public access", 1),
        ("Community center offers free computer classes for senior citizens", 1),
        ("Local business receives state award for outstanding sustainable practices", 1),
        ("University researchers develop new water purification method for rural areas", 1),
        ("Police department implements new community outreach program for youth", 1),
        ("Mayor announces plans for downtown revitalization project next year", 1),
        ("Hospital opens new emergency department wing to serve growing population", 1),
        ("School board approves construction of new elementary school building", 1),
        ("Local farmers market celebrates 25th anniversary with special events", 1),
    ]
    
    try:
        df = pd.DataFrame(sample_data, columns=['text', 'class'])
        df['processed_text'] = df['text'].apply(preprocess_text)
        df = df[df['processed_text'].str.len() > 0]
        
        if len(df) == 0:
            print("No valid data after preprocessing")
            return False
        
        print(f"Training with {len(df)} samples")
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['class'], test_size=0.2, random_state=30
        )
        
        vectorizer = TfidfVectorizer(
            max_features=1000,
            decode_error='replace',
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        x_train_tfidf = vectorizer.fit_transform(X_train)
        x_test_tfidf = vectorizer.transform(X_test)
        
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(x_train_tfidf, y_train)
        
        lr_pred = lr_model.predict(x_test_tfidf)
        accuracy = accuracy_score(y_test, lr_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        joblib.dump(lr_model, 'models/fake_news_model.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        print("Models trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error training models: {e}")
        return False

# URL Utils fallback functions
def is_valid_url(url):
    import re
    url_pattern = re.compile(
        r'^https?://'
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'
        r'localhost|'
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'
        r'(?::\d+)?'
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)
    return url_pattern.match(url) is not None

def extract_article_text(url):
    print("URL extraction not available - using fallback")
    return ""

def get_domain(url):
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    except:
        return "N/A"

def is_credible_domain(url):
    credible_domains = ['bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org', 'nytimes.com', 'washingtonpost.com']
    domain = get_domain(url)
    return any(credible in domain.lower() for credible in credible_domains)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Train models on startup
print("Setting up models...")
training_success = train_models_if_needed()

# Load models
lr_model = None
vectorizer = None

def load_models():
    global lr_model, vectorizer
    try:
        lr_model = joblib.load('models/fake_news_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        logger.info("Models loaded successfully.")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return False

models_loaded = load_models()

def improved_predict(text_vector):
    try:
        if lr_model is None:
            return 0, [0.5, 0.5], "No model available"
        
        if text_vector.shape[0] == 0 or text_vector.shape[1] == 0:
            return 0, [0.5, 0.5], "Empty input vector"
        
        proba = lr_model.predict_proba(text_vector)[0]
        threshold = 0.55
        prediction = 1 if proba[1] > threshold else 0
        
        return prediction, proba, "Prediction successful"
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0, [0.5, 0.5], f"Prediction error: {str(e)}"

@app.route('/')
def home():
    status_info = {
        'nltk': 'success' if nltk_setup_success else 'error',
        'training': 'success' if training_success else 'error',
        'models': 'success' if models_loaded else 'error'
    }
    return render_template('index.html', status=status_info)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        news_input = request.form.get('news_text', '').strip()
        
        if not news_input:
            result = {
                'prediction': 'ERROR',
                'confidence': 0,
                'text': 'No input provided',
                'fake_probability': 0,
                'real_probability': 0,
                'domain': 'N/A',
                'credible_domain': 'No',
                'debug_note': 'No input text provided',
                'status': 'error',
                'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            return render_template('result.html', result=result)
        
        if lr_model is None or vectorizer is None:
            result = {
                'prediction': 'MODEL ERROR',
                'confidence': 0,
                'text': news_input,
                'fake_probability': 50,
                'real_probability': 50,
                'domain': 'N/A',
                'credible_domain': 'No',
                'debug_note': 'Models not loaded properly',
                'status': 'error',
                'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            return render_template('result.html', result=result)
        
        # Process input
        is_url_input = is_valid_url(news_input)
        
        if is_url_input:
            article_text = extract_article_text(news_input)
            if not article_text or len(article_text.split()) < 5:
                result = {
                    'prediction': 'URL EXTRACTION FAILED',
                    'confidence': 0,
                    'text': news_input,
                    'fake_probability': 0,
                    'real_probability': 0,
                    'domain': get_domain(news_input),
                    'credible_domain': 'Yes' if is_credible_domain(news_input) else 'No',
                    'debug_note': 'Could not extract sufficient content from URL',
                    'status': 'warning',
                    'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
                }
                return render_template('result.html', result=result)
            
            processed_text = preprocess_text(article_text)
            domain = get_domain(news_input)
            is_credible = is_credible_domain(news_input)
        else:
            processed_text = preprocess_text(news_input)
            domain = "N/A"
            is_credible = False
        
        if not processed_text or len(processed_text.strip()) < 3:
            result = {
                'prediction': 'INSUFFICIENT TEXT',
                'confidence': 0,
                'text': news_input,
                'fake_probability': 0,
                'real_probability': 0,
                'domain': domain,
                'credible_domain': 'Yes' if is_credible else 'No',
                'debug_note': 'Processed text too short',
                'status': 'warning',
                'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            return render_template('result.html', result=result)
        
        # Vectorize and predict
        text_vector = vectorizer.transform([processed_text])
        prediction, proba, debug_msg = improved_predict(text_vector)
        
        # Determine status based on prediction
        if prediction == 1:
            status = 'real'
            prediction_text = 'REAL NEWS'
        else:
            status = 'fake'
            prediction_text = 'FAKE NEWS'
        
        result = {
            'prediction': prediction_text,
            'confidence': float(max(proba)) * 100,
            'text': news_input,
            'fake_probability': float(proba[0]) * 100,
            'real_probability': float(proba[1]) * 100,
            'domain': domain,
            'credible_domain': 'Yes' if is_credible else 'No',
            'debug_note': f'{debug_msg} | NLTK: {"OK" if nltk_setup_success else "Failed"}',
            'status': status,
            'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
        }
        
        return render_template('result.html', result=result)
        
    except Exception as e:
        result = {
            'prediction': 'CRITICAL ERROR',
            'confidence': 0,
            'text': news_input if 'news_input' in locals() else 'Unknown',
            'fake_probability': 0,
            'real_probability': 0,
            'domain': 'N/A',
            'credible_domain': 'No',
            'debug_note': f"Error: {str(e)}",
            'status': 'error',
            'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
        }
        return render_template('result.html', result=result)

if __name__ == '__main__':
    print(f"System Status:")
    print(f"- NLTK Setup: {'Success' if nltk_setup_success else 'Failed'}")
    print(f"- Training: {'Success' if training_success else 'Failed'}")
    print(f"- Models Loaded: {'Success' if models_loaded else 'Failed'}")
    app.run(debug=True, host='0.0.0.0', port=7860)

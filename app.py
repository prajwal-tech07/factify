import os
import sys
from flask import Flask, render_template, request
import joblib
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
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up NLTK data path for Hugging Face
nltk_data_dir = os.path.join(os.getcwd(), 'nltk_data')
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)
nltk.data.path.append(nltk_data_dir)

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
        nltk.download('punkt', download_dir=nltk_data_dir, quiet=True)
        nltk.download('punkt_tab', download_dir=nltk_data_dir, quiet=True)
        nltk.download('stopwords', download_dir=nltk_data_dir, quiet=True)
        nltk.download('wordnet', download_dir=nltk_data_dir, quiet=True)
        nltk.download('omw-1.4', download_dir=nltk_data_dir, quiet=True)
        print("NLTK resources downloaded successfully")
        return True
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")
        return False

nltk_setup_success = setup_nltk()

# Enhanced preprocessing functions
def clean_text(text):
    try:
        text = str(text).lower()
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Remove extra whitespace
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
            # Fallback to basic processing
            tokens = [word for word in text.split() if len(word) > 2 and word not in ['the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
        
        processed_text = ' '.join(tokens)
        return processed_text
        
    except Exception as e:
        print(f"Error in preprocess_text: {e}")
        return str(text) if text else ""

# Enhanced training function with more diverse data
def train_models_if_needed():
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Always retrain to ensure proper fitting
    print("Training models with enhanced dataset...")
    
    # Expanded training data with more diverse examples
    sample_data = [
        # Fake news examples (more diverse)
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
        ("Facebook will start charging users monthly fees unless you share this post", 0),
        ("Bill Gates admits putting microchips in COVID vaccines to track people", 0),
        ("Drinking bleach cures coronavirus according to secret government documents", 0),
        ("5G towers are actually mind control devices built by lizard people", 0),
        ("Scientists prove that the sun is actually cold and space is a hoax", 0),
        ("Breaking: All birds are government drones used for surveillance", 0),
        ("New research shows that wearing masks makes you grow gills", 0),
        ("Local man discovers that gravity is just a theory and starts floating", 0),
        ("Study reveals that social media likes can cure depression instantly", 0),
        ("Government secretly replaces all water with liquid that makes people obedient", 0),
        
        # Real news examples (more diverse)
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
        ("Federal Reserve maintains interest rates at current levels following economic review", 1),
        ("Scientists publish peer-reviewed study on climate change effects in Arctic regions", 1),
        ("International trade agreement signed between multiple countries to boost economic cooperation", 1),
        ("Health officials recommend updated vaccination schedules based on latest medical research", 1),
        ("Transportation department announces infrastructure improvements for highway safety", 1),
        ("Educational institutions report increased enrollment in STEM programs this semester", 1),
        ("Agricultural department releases guidelines for sustainable farming practices", 1),
        ("Technology sector shows steady growth with new job opportunities in emerging fields", 1),
        ("Medical researchers announce breakthrough in early cancer detection methods", 1),
        ("Environmental protection agency implements new regulations for water quality standards", 1),
    ]
    
    try:
        df = pd.DataFrame(sample_data, columns=['text', 'class'])
        print(f"Created dataset with {len(df)} samples")
        
        # Preprocess all texts
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Filter out empty processed texts
        df = df[df['processed_text'].str.len() > 0]
        
        if len(df) < 10:
            print("Insufficient data after preprocessing")
            return False
        
        print(f"Training with {len(df)} valid samples")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'], df['class'], test_size=0.2, random_state=42, stratify=df['class']
        )
        
        # Create and fit vectorizer with better parameters
        vectorizer = TfidfVectorizer(
            max_features=2000,
            decode_error='replace',
            stop_words='english',
            ngram_range=(1, 3),  # Include unigrams, bigrams, and trigrams
            min_df=1,
            max_df=0.95,
            lowercase=True,
            strip_accents='ascii'
        )
        
        # Fit and transform training data
        print("Fitting TF-IDF vectorizer...")
        x_train_tfidf = vectorizer.fit_transform(X_train)
        x_test_tfidf = vectorizer.transform(X_test)
        
        print(f"TF-IDF matrix shape: {x_train_tfidf.shape}")
        print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
        
        # Train the model
        lr_model = LogisticRegression(
            max_iter=2000, 
            random_state=42,
            C=1.0,
            class_weight='balanced'
        )
        
        print("Training logistic regression model...")
        lr_model.fit(x_train_tfidf, y_train)
        
        # Test the model
        lr_pred = lr_model.predict(x_test_tfidf)
        accuracy = accuracy_score(y_test, lr_pred)
        print(f"Model accuracy: {accuracy:.4f}")
        
        # Save the models
        print("Saving models...")
        joblib.dump(lr_model, 'models/fake_news_model.pkl')
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        print("Models trained and saved successfully!")
        return True
        
    except Exception as e:
        print(f"Error training models: {e}")
        import traceback
        traceback.print_exc()
        return False

# URL Utils functions
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
    # For now, return empty string - you can implement web scraping here
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
    credible_domains = [
        'bbc.com', 'cnn.com', 'reuters.com', 'ap.org', 'npr.org', 
        'nytimes.com', 'washingtonpost.com', 'theguardian.com',
        'wsj.com', 'bloomberg.com', 'abcnews.go.com', 'cbsnews.com'
    ]
    domain = get_domain(url)
    return any(credible in domain.lower() for credible in credible_domains)

# Flask app setup
app = Flask(__name__)

# Initialize models
print("Setting up models...")
training_success = train_models_if_needed()

lr_model = None
vectorizer = None

def load_models():
    global lr_model, vectorizer
    try:
        print("Loading models...")
        lr_model = joblib.load('models/fake_news_model.pkl')
        vectorizer = joblib.load('models/vectorizer.pkl')
        print("Models loaded successfully.")
        print(f"Vectorizer vocabulary size: {len(vectorizer.vocabulary_)}")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        import traceback
        traceback.print_exc()
        return False

models_loaded = load_models()

def improved_predict(text_vector):
    try:
        if lr_model is None:
            return 0, [0.5, 0.5], "No model available"
        
        if text_vector.shape[0] == 0 or text_vector.shape[1] == 0:
            return 0, [0.5, 0.5], "Empty input vector"
        
        # Get prediction probabilities
        proba = lr_model.predict_proba(text_vector)[0]
        
        # Use dynamic threshold based on confidence
        confidence_threshold = 0.6
        prediction = 1 if proba[1] > confidence_threshold else 0
        
        return prediction, proba, "Prediction successful"
        
    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback
        traceback.print_exc()
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
            # Try to reload models
            models_reloaded = load_models()
            if not models_reloaded:
                result = {
                    'prediction': 'MODEL ERROR',
                    'confidence': 0,
                    'text': news_input,
                    'fake_probability': 50,
                    'real_probability': 50,
                    'domain': 'N/A',
                    'credible_domain': 'No',
                    'debug_note': 'Models not loaded properly - please refresh the page',
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
                'debug_note': 'Processed text too short - please provide more detailed content',
                'status': 'warning',
                'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            return render_template('result.html', result=result)
        
        # Vectorize and predict
        try:
            print(f"Processing text: {processed_text[:100]}...")
            text_vector = vectorizer.transform([processed_text])
            print(f"Text vector shape: {text_vector.shape}")
            
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
            print(f"Vectorization error: {e}")
            import traceback
            traceback.print_exc()
            
            result = {
                'prediction': 'PROCESSING ERROR',
                'confidence': 0,
                'text': news_input,
                'fake_probability': 0,
                'real_probability': 0,
                'domain': domain,
                'credible_domain': 'Yes' if is_credible else 'No',
                'debug_note': f"Text processing error: {str(e)}",
                'status': 'error',
                'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
            }
            return render_template('result.html', result=result)
        
    except Exception as e:
        print(f"Critical error in predict: {e}")
        import traceback
        traceback.print_exc()
        
        result = {
            'prediction': 'CRITICAL ERROR',
            'confidence': 0,
            'text': news_input if 'news_input' in locals() else 'Unknown',
            'fake_probability': 0,
            'real_probability': 0,
            'domain': 'N/A',
            'credible_domain': 'No',
            'debug_note': f"Critical error: {str(e)}",
            'status': 'error',
            'timestamp': datetime.now().strftime('%B %d, %Y at %I:%M %p')
        }
        return render_template('result.html', result=result)

# Health check endpoint
@app.route('/health')
def health():
    return {
        'status': 'healthy', 
        'models_loaded': models_loaded,
        'nltk_setup': nltk_setup_success,
        'training_success': training_success
    }

if __name__ == '__main__':
    print(f"System Status:")
    print(f"- NLTK Setup: {'Success' if nltk_setup_success else 'Failed'}")
    print(f"- Training: {'Success' if training_success else 'Failed'}")
    print(f"- Models Loaded: {'Success' if models_loaded else 'Failed'}")
    
    port = int(os.environ.get('PORT', 7860))
    app.run(debug=False, host='0.0.0.0', port=port)

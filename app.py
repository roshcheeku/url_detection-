from flask import Flask, request, render_template, redirect, url_for, session
from joblib import load
import re
import sys
import types

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Ensure session works with Flask
app.config['SESSION_TYPE'] = 'filesystem'

# Define tokenizer function
def tokenizer(url):
    tokens = re.split('[/-]', url)
    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')
            if "com" in dot_split:
                dot_split.remove("com")
            if "www" in dot_split:
                dot_split.remove("www")
            tokens += dot_split
    # Log tokens
    print(f"Tokens: {tokens}")
    return tokens

# Custom load function to ensure tokenizer is available
def custom_load(file_name):
    # Create a new module to inject tokenizer into
    module_name = '__main__'
    module = types.ModuleType(module_name)
    sys.modules[module_name] = module

    # Inject the tokenizer function into the new module
    exec("""
def tokenizer(url):
    import re
    tokens = re.split('[/-]', url)
    for i in tokens:
        if i.find(".") >= 0:
            dot_split = i.split('.')
            if "com" in dot_split:
                dot_split.remove("com")
            if "www" in dot_split:
                dot_split.remove("www")
            tokens += dot_split
    return tokens
    """, module.__dict__)

    try:
        return load(file_name)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

# Load models and vectorizers using custom_load
cVec = custom_load('count_vectorizer.pkl')
tVec = custom_load('tfidf_vectorizer.pkl')
mnb_count = custom_load('mnb_count_model.pkl')
mnb_tfidf = custom_load('mnb_tfidf_model.pkl')
lgs_count = custom_load('lgs_count_model.pkl')
lgs_tfidf = custom_load('lgs_tfidf_model.pkl')

# Function to check URL and make predictions
def check_url(url):
    # Log the URL before processing
    print(f"Processing URL: {url}")
    
    if not cVec or not tVec or not mnb_count or not mnb_tfidf or not lgs_count or not lgs_tfidf:
        raise Exception("Model loading failed. Please check the model files.")
    
    count_vec = cVec.transform([url])
    tfidf_vec = tVec.transform([url])

    print("Count Vectorizer Output:", count_vec)
    print("TF-IDF Vectorizer Output:", tfidf_vec)

    pred_mnb_count = mnb_count.predict(count_vec)[0]
    pred_mnb_tfidf = mnb_tfidf.predict(tfidf_vec)[0]
    pred_lgs_count = lgs_count.predict(count_vec)[0]
    pred_lgs_tfidf = lgs_tfidf.predict(tfidf_vec)[0]

    predictions = {
        "Multinomial Naive Bayes (Count Vectorizer)": pred_mnb_count,
        "Multinomial Naive Bayes (TF-IDF)": pred_mnb_tfidf,
        "Logistic Regression (Count Vectorizer)": pred_lgs_count,
        "Logistic Regression (TF-IDF)": pred_lgs_tfidf
    }
    print(f"Predictions: {predictions}")

    return predictions

# Route to render home page
@app.route('/')
def home():
    return redirect(url_for('login'))

# Route to render login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Simulate successful login
        return redirect(url_for('index'))
    return render_template('login.html')

# Route to render index page after login
@app.route('/index')
def index():
    search_history = session.get('search_history', [])
    return render_template('home.html', search_history=search_history)

# Route to render help page
@app.route('/help')
def help():
    return render_template('help.html')

# Route to render about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route to render profile page
@app.route('/profile')
def profile():
    return render_template('profile.html')
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        # Add logic here to handle registration (e.g., save to database)
        return redirect(url_for('login'))  # Redirect to login page after registration
    return render_template('register.html')

# Route to handle URL predictions
# Route to handle URL predictions
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form['url']
    
    # Log the received URL
    print(f"Received URL: {url}")
    
    if not url or not re.match(r'https?://.*', url):
        return "Invalid URL", 400
    
    try:
        predictions = check_url(url)
    except Exception as e:
        return f"Error: {e}", 500

    # Update search history
    search_history = session.get('search_history', [])
    if url not in search_history:
        search_history.append(url)
        session['search_history'] = search_history

    return render_template('result.html', url=url, results=predictions)

# Route to handle feedback
@app.route('/send_feedback', methods=['POST'])
def send_feedback():
    feedback = request.json.get('feedback')
    if feedback:
        # Handle the feedback (e.g., store in database or send an email)
        print(f"Feedback received: {feedback}")
        return 'Feedback sent successfully', 200
    return 'No feedback provided', 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)

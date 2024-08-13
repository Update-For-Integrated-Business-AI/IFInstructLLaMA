# Import necessary libraries
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from langdetect import detect

# Load data
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# Detect language
def detect_language(text):
    try:
        return detect(text)
    except:
        return None

"""# Preprocessing
def preprocess_data(data):
    # Initialize the lemmatizer and define stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    processed_data = []
    for document in data:
        # Detect the language of the document
        language = detect_language(document)
        if language == 'en':
            # Tokenize the document
            tokens = word_tokenize(document)
            # Apply lemmatization
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            # Remove stop words
            tokens = [word for word in tokens if word not in stop_words]
            processed_data.append(' '.join(tokens))

    return processed_data"""
# Import necessary libraries
from nltk.stem import SnowballStemmer

# Preprocessing
def preprocess_data(data):
    # Initialize the lemmatizer and stemmer
    lemmatizer = WordNetLemmatizer()
    stemmer = SnowballStemmer("arabic")

    # Define stop words for both English and Arabic
    english_stop_words = set(stopwords.words('english'))
    arabic_stop_words = set(stopwords.words('arabic'))

    processed_data = []
    for document in data:
        # Detect the language of the document
        document = str(document)
        language = detect_language(document)
        if language == 'en':
            # Tokenize the document
            tokens = word_tokenize(document)
            # Apply lemmatization
            tokens = [lemmatizer.lemmatize(token) for token in tokens]
            # Remove English stop words
            tokens = [word for word in tokens if word not in english_stop_words]
        else: #language == 'ar':
            # Tokenize the document
            tokens = word_tokenize(document)
            # Apply stemming
            tokens = [stemmer.stem(token) for token in tokens]
            # Remove Arabic stop words
            tokens = [word for word in tokens if word not in arabic_stop_words]
        #print(document, language)
        processed_data.append(' '.join(tokens))

    return processed_data


# Feature Extraction
def extract_features(processed_data):
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    counts = vectorizer.fit_transform(processed_data)
    features = tfidf_transformer.fit_transform(counts)

    return features, vectorizer, tfidf_transformer

# Model Training
def train_model_NB(features, labels):
    model = MultinomialNB()
    print(labels[0], "asdasda")
    model.fit(features, labels)
    return model
from sklearn.svm import SVC

# Model Training - Updated for SVM
def train_model_SVM(features, labels):
    model = SVC(kernel='linear')  # You can change the kernel and other parameters
    model.fit(features, labels)
    return model
from sklearn.ensemble import RandomForestClassifier
# Model Training - Updated for Random Forest
def train_model_randomforest(features, labels):
    model = RandomForestClassifier(n_estimators=100, random_state=0)  # You can adjust parameters like n_estimators
    model.fit(features, labels)
    return model
from sklearn.neighbors import KNeighborsClassifier
def train_model_knn(features, labels):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(features, labels)
    return model
from sklearn.ensemble import GradientBoostingClassifier
def train_model_GradientBoostingClassifier(features, labels):
    model = GradientBoostingClassifier( random_state=0)
    model.fit(features, labels)
    return model
# Evaluation
def evaluate_model(model, test_features, test_labels):
    predictions = model.predict(test_features)
    precision = precision_score(test_labels, predictions, average='weighted')
    recall = recall_score(test_labels, predictions, average='weighted')
    f1 = f1_score(test_labels, predictions, average='weighted')
    accuracy = accuracy_score(test_labels, predictions)

    return precision, recall, f1, accuracy

# Main function
def main():
    # Load data
    train_val = load_data('train_val.csv')

    # Check for NaN values in labels
    if pd.isna(train_val['class']).any():
        # Handle NaN values (remove or impute)
        train_val = train_val.dropna(subset=['class'])

    # Preprocess data
    processed_data_train_val = preprocess_data(train_val['Item_Name'])

    # Extract features
    train_features, vectorizer, tfidf_transformer = extract_features(processed_data_train_val)
    print(train_val['class'][0])
    # Split data into training and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(train_features, train_val['class'], test_size=0.2)

    # Train model
    model = train_model_SVM(train_features, train_labels)

    # Evaluate model
    precision, recall, f1, accuracy = evaluate_model(model, test_features, test_labels)

    # Print evaluation metrics
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Accuracy: {accuracy}')

    test = load_data('test.csv')
    if pd.isna(test['class']).any():
        # Handle NaN values (remove or impute)
        test = test.dropna(subset=['class'])
    processed_data_test = preprocess_data(test['Item_Name'])
    # Transform test data using the same vectorizer and tfidf_transformer
    test_counts = vectorizer.transform(processed_data_test)
    test_features = tfidf_transformer.transform(test_counts)
    # Evaluate model
    precision, recall, f1, accuracy = evaluate_model(model, test_features, test['class'])
    # Print evaluation metrics
    print(f'Precision test: {precision*100}')
    print(f'Recall test: {recall*100}')
    print(f'F1 Score test: {f1*100}')
    print(f'Accuracy test: {accuracy*100}')

if __name__ == "__main__":
    main()

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem.porter import PorterStemmer
from bs4 import BeautifulSoup
import re
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from wordcloud import WordCloud
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')
nltk.download('stopwords')
nltk.download('punkt')

def strip_html(text):
    return BeautifulSoup(text, "html.parser").get_text()

def remove_between_square_brackets(text):
    return re.sub(r'\[[^]]*\]', '', text)

def denoise_text(text):
    return remove_between_square_brackets(strip_html(text))

def remove_special_characters(text):
    return re.sub(r'[^a-zA-Z0-9\s]', '', text)

def simple_stemmer(text):
    ps = PorterStemmer()
    return ' '.join([ps.stem(word) for word in text.split()])

def remove_stopwords(text, tokenizer, stopword_list):
    tokens = tokenizer.tokenize(text)
    return ' '.join([token for token in tokens if token.lower() not in stopword_list])

def plot_wordcloud(text, title):
    plt.figure(figsize=(10, 6))
    wc = WordCloud(width=1000, height=500, max_words=200).generate(text)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()

# Convert review text to sentiment label using TextBlob
def get_sentiment_label(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def main():
    print("Loading dataset...")
   
    imdb_data = pd.read_csv('C:/Users/acer/Desktop/Sentiment/imdb_top_1000.csv')
    print("Columns in dataset:", imdb_data.columns.tolist())  
    

    print("Generating new sentiment labels (positive/negative/neutral)...")
    imdb_data['sentiment'] = imdb_data['Overview'].apply(get_sentiment_label)

    print(imdb_data['sentiment'].value_counts())

    # Preprocess
    imdb_data['Overview'] = imdb_data['Overview'].apply(denoise_text)
    imdb_data['Overview'] = imdb_data['Overview'].apply(remove_special_characters)
    tokenizer = ToktokTokenizer()
    stopword_list = stopwords.words('english')

    imdb_data['Overview'] = imdb_data['Overview'].apply(simple_stemmer)
    imdb_data['Overview'] = imdb_data['Overview'].apply(lambda x: remove_stopwords(x, tokenizer, stopword_list))
    # Encode labels
    le = LabelEncoder()
    imdb_data['sentiment_encoded'] = le.fit_transform(imdb_data['sentiment'])  # e.g., 0: negative, 1: neutral, 2: positive

    # Train/test split
    # train = imdb_data[:40000]
    # test = imdb_data[40000:]

    # X_train = train['Overview']
    # X_test = test['Overview']

    # y_train = train['sentiment_encoded']
    # y_test = test['sentiment_encoded']

    from sklearn.model_selection import train_test_split

    X = imdb_data['Overview']
    y = imdb_data['sentiment_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
     )


    # Vectorize
    cv = CountVectorizer(ngram_range=(1, 2))
    tv = TfidfVectorizer(ngram_range=(1, 2))

    cv_train = cv.fit_transform(X_train)
    cv_test = cv.transform(X_test)

    tv_train = tv.fit_transform(X_train)
    tv_test = tv.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=500),
        "Linear SVM": SGDClassifier(loss='hinge', max_iter=500, random_state=42),
        "Multinomial NB": MultinomialNB()
    }

    for name, model in models.items():
        print(f"\n{name} - CountVectorizer")
        model.fit(cv_train, y_train)
        pred_cv = model.predict(cv_test)
        print("Accuracy:", accuracy_score(y_test, pred_cv))
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred_cv))
        print("Classification Report:\n", classification_report(y_test, pred_cv, target_names=le.classes_))

        print(f"\n{name} - TF-IDF")
        model.fit(tv_train, y_train)
        pred_tv = model.predict(tv_test)
        print("Accuracy:", accuracy_score(y_test, pred_tv))
        print("Confusion Matrix:\n", confusion_matrix(y_test, pred_tv))
        print("Classification Report:\n", classification_report(y_test, pred_tv, target_names=le.classes_))

    # WordClouds (optional)
    print("\nGenerating WordClouds...")
    for label in le.classes_:
       text_blob = ' '.join(imdb_data[imdb_data['sentiment'] == label]['Overview'].head(100))

       plot_wordcloud(text_blob, f"WordCloud for {label.capitalize()} Reviews")

if __name__ == "__main__":
    main()

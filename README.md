# Micro_IIT_Internship_NoorySaba_Sentiment_Analysis-


# 🎬 Sentiment Analysis on IMDb Top 1000 Movies

This project performs sentiment analysis on movie overviews from the IMDb Top 1000 dataset using machine learning models and natural language processing (NLP).

---

## 📊 Project Objective

To classify each movie overview as:
- 👍 **Positive**
- 👎 **Negative**
- 😐 **Neutral**

Sentiments are automatically labeled using **TextBlob** polarity scores, and various models are trained to predict these sentiments.

---

## 📁 Files Included

- `main.py` – Python script for preprocessing, training, evaluation, and visualization.
- `imdb_top_1000.csv` – Dataset containing IMDb movie details including overviews.

---

## 🛠️ Tools & Libraries Used

- **Python** 🐍  
- **TextBlob** – for sentiment labeling  
- **Scikit-learn** – for ML models  
- **NLTK** – for tokenizing, stopwords, and stemming  
- **BeautifulSoup** – for HTML cleanup  
- **WordCloud** – to visualize word distributions  
- **CountVectorizer & TfidfVectorizer** – for feature extraction  

---

## 🔄 Process Flow

1. **Load Dataset**  
2. **Clean Text**  
   - Remove HTML tags  
   - Remove special characters  
   - Apply stemming  
   - Remove stopwords  
3. **Sentiment Labeling** with TextBlob  
4. **Vectorization**  
   - CountVectorizer  
   - TF-IDF  
5. **Train & Evaluate Models**
   - Logistic Regression  
   - Linear SVM  
   - Multinomial Naive Bayes  
6. **Generate Word Clouds** for each sentiment

---

## ▶️ How to Run

Make sure you have the required packages installed.

## 📌 Requirements
Make sure you have the following Python libraries installed:

pip install numpy pandas seaborn matplotlib scikit-learn textblob nltk wordcloud beautifulsoup4

```bash
pip install -r requirements.txt
python main.py



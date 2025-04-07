# Micro_IIT_Internship_NoorySaba_Sentiment_Analysis-


# Sentiment Analysis on IMDb Movies 🎬

This project analyzes the **sentiment** of movie overviews from IMDb’s Top 1000 Movies.  
It uses **machine learning** and **text processing** to classify each overview as:

- 👍 Positive  
- 👎 Negative  
- 😐 Neutral  

## 📁 Files

- `main.py` – Main Python script for sentiment analysis  
- `imdb_top_1000.csv` – Dataset with movie details

## 🛠 Tools Used

- Python
- pandas, numpy, scikit-learn
- TextBlob (for sentiment labeling)
- CountVectorizer & TF-IDF
- WordCloud for visualizations

## Steps

1. Clean and prepare the movie overview text
2. Label sentiment using TextBlob
3. Train models like:
   - Logistic Regression
   - Naive Bayes
   - SVM
4. Show results and word clouds

## ▶️ Run the Project

##bash
python main.py

Output
Accuracy for each model

Word clouds for Positive, Negative, Neutral reviews


## 📌 Requirements

Make sure you have the following Python libraries installed:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn textblob nltk wordcloud beautifulsoup4

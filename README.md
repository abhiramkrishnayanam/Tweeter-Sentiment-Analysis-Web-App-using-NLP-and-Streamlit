# Tweeter-Sentiment-Analysis-using-NLP
Using NLP techniques, I have built a web app for Sentiment Analysis
# Sentiment Analysis Web App

## Overview
This Sentiment Analysis Web App analyzes tweets to classify their sentiment as positive, negative, or neutral. The project includes data preprocessing, model building, and deployment using Streamlit.

## Dataset
- The dataset used consists of tweets with labeled sentiments.
- Data was cleaned and explored through Exploratory Data Analysis (EDA) to understand patterns and trends.
- Missing values were handled, and data distribution was analyzed for better insights.


![image](https://github.com/user-attachments/assets/b64161ad-2804-45a1-880a-818e472be41e)


## NLP Preprocessing Steps
To ensure high-quality text data for modeling, the following preprocessing steps were applied:
- **Tokenization**: Breaking down text into individual words or tokens.
- **Removing Stopwords**: Eliminating common words that do not contribute to sentiment, such as 'is', 'the', 'and'.
- **Stemming**: Reducing words to their root forms using algorithms like Porter Stemmer.
- **Lemmatization**: Converting words to their base or dictionary form (e.g., 'running' to 'run').
- **Regular Expressions (Regex) for text cleaning**: Removing special characters, punctuation, and URLs.
- **N-Grams**: Creating word sequences (bigrams, trigrams) to capture phrase-level sentiment.
- **Part-of-Speech (POS) Tagging**: Identifying word types such as nouns, verbs, and adjectives to understand context.
- **Named Entity Recognition (NER)**: Extracting named entities such as names, locations, and brands.
- **Bag of Words (BoW)**: Converting text into numerical representation using word frequency.
- **TF-IDF Model**: Transforming text data into numerical format by assigning importance weights to words.

## Model Building
Three different models were implemented to classify the sentiment:
- **Logistic Regression**: A simple and effective model for binary and multiclass classification.
- **Naïve Bayes**: A probabilistic classifier based on Bayes' theorem, well-suited for text classification.
- **Artificial Neural Networks (ANN)**: A deep learning-based approach to improve model accuracy.

### Model Training and Evaluation
- Each model was trained using a preprocessed dataset.
- Performance was evaluated using accuracy, precision, recall, and F1-score.
- Hyperparameter tuning was performed to optimize the best model.
- The best-performing model was selected based on evaluation metrics.
- Final predictions were generated and compared for accuracy.

## Model Deployment
- Saved the trained models using **joblib** or **pickle** for future predictions.
- Developed a **Streamlit** web app (`app.py`) to provide an interactive user interface.
- Integrated **CSS** for styling and an enhanced user experience.
- Deployed the web app on **Streamlit Cloud** for public access.

## How to Run Locally
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/sentiment-analysis-app.git
   cd sentiment-analysis-app
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn, TensorFlow/Keras
- NLTK, spaCy
- Streamlit
- CSS

## Future Improvements
- Improve model accuracy using more advanced deep learning architectures.
- Add real-time tweet analysis feature.
- Integrate with social media APIs for live data fetching.
- Expand to multilingual sentiment analysis.

## Contact
For any queries, feel free to reach out at [your-email@example.com](mailto:your-email@example.com).

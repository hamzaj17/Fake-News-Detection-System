# 📰 Fake News Detection System (ML + NLP + Streamlit)

A web-based Fake News Detection system built with Python, Streamlit, and Machine Learning (Logistic Regression) that classifies news articles as **TRUE** or **FAKE**.

This project uses the `True.csv` and `Fake.csv` datasets, applies text preprocessing (cleaning), TF-IDF vectorization, and trains a Logistic Regression classifier. Additionally, Rocchio classification with cosine similarity is implemented for a secondary prediction method. 

- My Fake News Detection system has a model accuracy of `98.28%`.

The app offers the following features:
- 📄 **Classify Text**: Paste any article to check if it's fake or real.
- 🌐 **Classify from URL**: Enter a news article URL and automatically fetch and classify it.
- 🔍 **Search News**: Search dataset for a keyword and explore matching news articles.

## 🚀 Technologies Used
- Python
- Streamlit
- Pandas, NumPy
- Scikit-learn (Logistic Regression, TF-IDF)
- Newspaper3k (for URL scraping)
- Cosine Similarity (Rocchio-style classification)

## 📁 Dataset Source
- `Fake.csv` and `True.csv` from the Kaggle Fake News dataset (manually downloaded).
  
 <p>link: https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset</p>

 ---

## 🛠️ How to Run

1. Clone the repository  
   `git clone https://github.com/hamzaj17/Fake-News-Detection-System.git`

2. Navigate to project folder  
   `cd Fake-News-Detection-System`

3. Install the required libraries  

4. Run the Streamlit app  
   `streamlit run app.py`

---

## 🙋‍♂️ Author
Hamza Bin Javed  
[LinkedIn](https://www.linkedin.com/in/hamzaj17) | [GitHub](https://github.com/hamzaj17)

---

## 📄 License
This project is open source under the MIT License.

# 🎬 Movie Recommendation System

## 📌 Overview

A scalable Movie Recommendation System implementing **collaborative filtering**, **content-based filtering**, and **semantic similarity using transformer embeddings**. The system delivers personalized recommendations through similarity metrics and an interactive Streamlit interface.

---

## 🚀 Key Features

* 👥 **Collaborative Filtering** using user–user similarity
* 📄 **Content-Based Filtering** using TF-IDF on movie overviews
* 🧠 **Semantic Recommendations** using Sentence Transformers (MiniLM)
* 🔍 **Fuzzy Matching** to handle incorrect or partial user input
* ⚡ **Optimized with caching** for faster inference
* 🎯 Real-time recommendations via Streamlit UI

---

## 🧠 Methodology

### 1. Collaborative Filtering

* Built a user–movie interaction matrix
* Computed similarity using cosine similarity
* Recommended movies based on similar users’ preferences

### 2. Content-Based Filtering

* Converted movie overviews into TF-IDF vectors
* Used cosine similarity to find similar movies

### 3. Semantic Recommendation

* Generated embeddings using transformer model (`all-MiniLM-L6-v2`)
* Captured semantic meaning beyond keywords
* Recommended movies based on embedding similarity

---

## 🛠️ Tech Stack

* **Python**
* **Pandas, NumPy**
* **Scikit-learn**
* **Sentence Transformers**
* **Streamlit**

---

## ⚡ Performance Optimizations

* Cached TF-IDF matrix and models using Streamlit caching
* Reduced redundant computations during inference

---

## 🧩 Challenges & Solutions

* ❌ *Movie not found issue* → ✅ Fixed using fuzzy matching (`difflib`)
* ❌ *Model input errors* → ✅ Resolved by converting data formats properly
* ❌ *Slow execution* → ✅ Improved using caching

---

## 📂 Project Structure

```
Movie-Recommendation-System/
│
├── app.py
├── collaborative.py
├── content_based.py
├── embeddings.py
├── requirements.txt
├── README.md
└── Data/
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📌 Future Enhancements

* Hybrid recommendation system (combine all approaches)
* Integration of movie posters & metadata
* Scalable recommendation using matrix factorization (SVD/ALS)

---

## 📊 Outcome

Successfully built a multi-model recommendation system capable of delivering personalized movie suggestions with improved accuracy using semantic understanding.

---

# Technology-Stack
A Python-based Jewelry Analysis System using ML, deep learning, and NLP to predict prices, analyze reviews, visualize trends, and extract customer sentiment, enabling data-driven insights for pricing, recommendations, and trend forecasting in the jewelry domain.
# 💎 Jewelry Product Analysis

This project analyzes a jewelry product dataset using **Python** and multiple data science libraries.
It includes **data handling, visualization, machine learning, NLP sentiment analysis, and deep learning.**

---

## 📂 Dataset

If no dataset is provided, the script automatically generates a mock dataset named `jewelry_dataset.csv`.

**Fields:**
- Product_ID  
- Product_Name  
- Category  
- Price  
- Discount  
- Material  
- Polish  
- Rating  
- Review_Text  

---

## ⚙️ Technology Stack

| Category | Tools/Libraries Used |
|-----------|----------------------|
| Data Handling | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Matplotlib, Seaborn |
| NLP | NLTK, spaCy, Transformers |
| Deep Learning | TensorFlow, PyTorch |

---

## 🧠 Models Trained

| Model Type | Framework | Output File |
|-------------|------------|--------------|
| Linear Regression | Scikit-learn | `price_predictor.pkl` |
| Neural Network | TensorFlow | `tf_price_predictor.h5` |
| Neural Network | PyTorch | `torch_price_predictor.pth` |

---

## 🪄 Features

- Automatic dataset generation if not provided  
- Price prediction model (ML)  
- Sentiment analysis using NLTK and Transformers  
- Deep learning model examples (TensorFlow + PyTorch)  
- Data visualization (Seaborn + Matplotlib)

---

## ▶️ How to Run

### 1️⃣ Install Requirements
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk spacy transformers tensorflow torch joblib
python -m spacy download en_core_web_sm

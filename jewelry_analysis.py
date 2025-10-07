# -------------------------------
# 💎 Jewelry Product Analysis System
# -------------------------------

# Core Libraries
import pandas as pd
import numpy as np

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from transformers import pipeline

# Deep Learning
import tensorflow as tf
import torch
import torch.nn as nn

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------
# Step 1: Create Mock Jewelry Dataset
# --------------------------------------

np.random.seed(42)
n = 100

data = {
    "Product_ID": range(1, n+1),
    "Product_Name": [f"Jewelry_{i}" for i in range(1, n+1)],
    "Category": np.random.choice(["Necklace", "Ring", "Bracelet", "Earring"], n),
    "Price": np.random.randint(1000, 10000, n),
    "Discount": np.random.randint(5, 30, n),
    "Material": np.random.choice(["Gold", "Silver", "Platinum", "Diamond"], n),
    "Polish": np.random.choice(["Matte", "Glossy", "High Polish"], n),
    "Rating": np.round(np.random.uniform(2.5, 5.0, n), 1),
    "Review_Text": np.random.choice([
        "Absolutely beautiful piece!",
        "Good quality but a bit overpriced.",
        "Excellent craftsmanship!",
        "Not satisfied, tarnished quickly.",
        "Loved it! Looks premium."
    ], n)
}

df = pd.DataFrame(data)
print("✅ Dataset Created Successfully!\n")
print(df.head())

# --------------------------------------
# Step 2: Data Analysis
# --------------------------------------
print("\n--- Basic Info ---")
print(df.info())
print("\n--- Summary Statistics ---")
print(df.describe())

# --------------------------------------
# Step 3: Data Visualization
# --------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="Category", y="Price", data=df, estimator=np.mean)
plt.title("Average Price by Jewelry Category")
plt.show()

plt.figure(figsize=(6, 4))
sns.histplot(df["Rating"], bins=10, kde=True)
plt.title("Distribution of Ratings")
plt.show()

# --------------------------------------
# Step 4: Machine Learning - Price Prediction
# --------------------------------------

# Encode categorical variables
df_encoded = pd.get_dummies(df[["Category", "Material", "Polish"]], drop_first=True)
X = pd.concat([df_encoded, df[["Discount", "Rating"]]], axis=1)
y = df["Price"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("\n--- ML Model Evaluation ---")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

# --------------------------------------
# Step 5: NLP - Sentiment Analysis on Reviews
# --------------------------------------
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()
df["Sentiment_Score"] = df["Review_Text"].apply(lambda x: sia.polarity_scores(x)["compound"])

plt.figure(figsize=(6, 4))
sns.boxplot(x="Category", y="Sentiment_Score", data=df)
plt.title("Sentiment Score by Jewelry Category")
plt.show()

# Using spaCy for text preprocessing
nlp = spacy.load("en_core_web_sm")
df["Cleaned_Review"] = df["Review_Text"].apply(lambda x: " ".join([token.lemma_ for token in nlp(x.lower()) if not token.is_stop]))

print("\nSample Cleaned Review Texts:")
print(df["Cleaned_Review"].head())

# Transformers (HuggingFace) for deep NLP
sentiment_pipeline = pipeline("sentiment-analysis")
sample_reviews = df["Review_Text"][:3].tolist()
print("\n--- Transformer Sentiment Results ---")
for review in sample_reviews:
    print(review, "->", sentiment_pipeline(review))

# --------------------------------------
# Step 6: Deep Learning Demonstration
# --------------------------------------

# TensorFlow (simple model)
tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1)
])
tf_model.compile(optimizer='adam', loss='mse')
tf_model.fit(X_train, y_train, epochs=3, verbose=1)
print("\n✅ TensorFlow Model Trained Successfully!")

# PyTorch (simple regression model)
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

torch_model = SimpleNN(X.shape[1])
print("\n✅ PyTorch Model Initialized Successfully!")

# --------------------------------------
# Step 7: Final Output
# --------------------------------------
print("\n--- Final Jewelry Dataset with Sentiments ---")
print(df.head())

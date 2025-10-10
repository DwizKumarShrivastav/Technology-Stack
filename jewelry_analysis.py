# -------------------------------
# 💎 Jewelry Product Analysis System (Final Version)
# -------------------------------

# Core Libraries
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Deep Learning
import tensorflow as tf
import torch
import torch.nn as nn

# --------------------------------------
# Step 1: Dataset Creation
# --------------------------------------

np.random.seed(42)
n = 250

data = {
    "Product_ID": range(1, n + 1),
    "Product_Name": [f"Jewelry_{i}" for i in range(1, n + 1)],
    "Category": np.random.choice(["Necklace", "Ring", "Bracelet", "Earring", "Mangalsutra"], n),
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
os.makedirs("data", exist_ok=True)
df.to_csv("data/jewelry_dataset.csv", index=False)
print("✅ jewelry_dataset.csv saved successfully!\n")

# --------------------------------------
# Step 2: Price Prediction (Regression)
# --------------------------------------

df_encoded = pd.get_dummies(df[["Category", "Material", "Polish"]], drop_first=True)
X = pd.concat([df_encoded, df[["Discount", "Rating"]]], axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(reg_model, "models/price_predictor.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

y_pred = reg_model.predict(X_test)
print("✅ Price Prediction Model Trained and Saved\n")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}\n")

# Visualization: Regression Results
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Jewelry Prices")
plt.show()

# --------------------------------------
# Step 3: Product Categorization (Classification)
# --------------------------------------

cat_X = pd.get_dummies(df[["Material", "Polish"]], drop_first=True)
cat_X = pd.concat([cat_X, df[["Price", "Discount", "Rating"]]], axis=1)
cat_y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(cat_X, cat_y, test_size=0.2, random_state=42)

cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
cat_model.fit(X_train, y_train)
pred_cat = cat_model.predict(X_test)

acc = accuracy_score(y_test, pred_cat)
print(f"✅ Product Categorization Accuracy: {acc * 100:.2f}%\n")

joblib.dump(cat_model, "models/category_classifier.pkl")

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, pred_cat, labels=cat_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Jewelry Category Prediction")
plt.show()

# --------------------------------------
# Step 4: Recommendation System
# --------------------------------------

features = pd.get_dummies(df[["Material", "Category", "Polish"]], drop_first=True)
recommender = NearestNeighbors(metric="cosine", algorithm="brute")
recommender.fit(features)

def recommend_products(product_name, n_recommendations=5):
    product = df[df["Product_Name"] == product_name]
    if product.empty:
        print("❌ Product not found!")
        return
    idx = product.index[0]
    distances, indices = recommender.kneighbors(features.iloc[[idx]], n_neighbors=n_recommendations + 1)
    similar_indices = indices.flatten()[1:]
    recommendations = df.iloc[similar_indices][["Product_Name", "Category", "Material", "Price", "Rating"]]
    print(f"\n💍 Recommended products similar to {product_name}:")
    print(recommendations)

recommend_products("Jewelry_10")

# --------------------------------------
# Step 5: Sentiment Analysis (NLP)
# --------------------------------------

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
df["Sentiment_Score"] = df["Review_Text"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Sentiment_Label"] = df["Sentiment_Score"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))

plt.figure(figsize=(6, 4))
sns.countplot(x="Sentiment_Label", data=df, palette="pastel")
plt.title("Customer Sentiment Distribution")
plt.show()

print("\n✅ Sentiment Analysis Completed!")

# --------------------------------------
# Step 6: Deep Learning (Price Prediction)
# --------------------------------------

tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])
tf_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = tf_model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.2)

tf_model.save("models/price_predictor_tf.h5")
print("✅ TensorFlow Model Trained and Saved!\n")

# Visualization: Loss Curve
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("TensorFlow Model Training Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------
# Step 7: Output Summary
# --------------------------------------

print("\n✅ All Tasks Completed Successfully!")
print("📂 Models saved in 'models/' folder")
print("📊 Dataset saved in 'data/' folder")
print("📈 Visualizations generated above\n")

print("--- Final Jewelry Dataset with Sentiments ---")
print(df.head())
# -------------------------------
# 💎 Jewelry Product Analysis System (Final Version)
# -------------------------------

# Core Libraries
import pandas as pd
import numpy as np
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import NearestNeighbors

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# NLP
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import spacy

# Deep Learning
import tensorflow as tf
import torch
import torch.nn as nn

# --------------------------------------
# Step 1: Dataset Creation
# --------------------------------------

np.random.seed(42)
n = 250

data = {
    "Product_ID": range(1, n + 1),
    "Product_Name": [f"Jewelry_{i}" for i in range(1, n + 1)],
    "Category": np.random.choice(["Necklace", "Ring", "Bracelet", "Earring", "Mangalsutra"], n),
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
os.makedirs("data", exist_ok=True)
df.to_csv("data/jewelry_dataset.csv", index=False)
print("✅ jewelry_dataset.csv saved successfully!\n")

# --------------------------------------
# Step 2: Price Prediction (Regression)
# --------------------------------------

df_encoded = pd.get_dummies(df[["Category", "Material", "Polish"]], drop_first=True)
X = pd.concat([df_encoded, df[["Discount", "Rating"]]], axis=1)
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

os.makedirs("models", exist_ok=True)
joblib.dump(reg_model, "models/price_predictor.pkl")
joblib.dump(X.columns.tolist(), "models/model_columns.pkl")

y_pred = reg_model.predict(X_test)
print("✅ Price Prediction Model Trained and Saved\n")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}\n")

# Visualization: Regression Results
plt.figure(figsize=(6, 4))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Jewelry Prices")
plt.show()

# --------------------------------------
# Step 3: Product Categorization (Classification)
# --------------------------------------

cat_X = pd.get_dummies(df[["Material", "Polish"]], drop_first=True)
cat_X = pd.concat([cat_X, df[["Price", "Discount", "Rating"]]], axis=1)
cat_y = df["Category"]

X_train, X_test, y_train, y_test = train_test_split(cat_X, cat_y, test_size=0.2, random_state=42)

cat_model = RandomForestClassifier(n_estimators=100, random_state=42)
cat_model.fit(X_train, y_train)
pred_cat = cat_model.predict(X_test)

acc = accuracy_score(y_test, pred_cat)
print(f"✅ Product Categorization Accuracy: {acc * 100:.2f}%\n")

joblib.dump(cat_model, "models/category_classifier.pkl")

# Confusion Matrix Visualization
cm = confusion_matrix(y_test, pred_cat, labels=cat_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=cat_model.classes_)
disp.plot(cmap="Blues")
plt.title("Confusion Matrix - Jewelry Category Prediction")
plt.show()

# --------------------------------------
# Step 4: Recommendation System
# --------------------------------------

features = pd.get_dummies(df[["Material", "Category", "Polish"]], drop_first=True)
recommender = NearestNeighbors(metric="cosine", algorithm="brute")
recommender.fit(features)

def recommend_products(product_name, n_recommendations=5):
    product = df[df["Product_Name"] == product_name]
    if product.empty:
        print("❌ Product not found!")
        return
    idx = product.index[0]
    distances, indices = recommender.kneighbors(features.iloc[[idx]], n_neighbors=n_recommendations + 1)
    similar_indices = indices.flatten()[1:]
    recommendations = df.iloc[similar_indices][["Product_Name", "Category", "Material", "Price", "Rating"]]
    print(f"\n💍 Recommended products similar to {product_name}:")
    print(recommendations)

recommend_products("Jewelry_10")

# --------------------------------------
# Step 5: Sentiment Analysis (NLP)
# --------------------------------------

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()
df["Sentiment_Score"] = df["Review_Text"].apply(lambda x: sia.polarity_scores(x)["compound"])
df["Sentiment_Label"] = df["Sentiment_Score"].apply(lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral"))

plt.figure(figsize=(6, 4))
sns.countplot(x="Sentiment_Label", data=df, palette="pastel")
plt.title("Customer Sentiment Distribution")
plt.show()

print("\n✅ Sentiment Analysis Completed!")

# --------------------------------------
# Step 6: Deep Learning (Price Prediction)
# --------------------------------------

tf_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(X.shape[1],)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1)
])
tf_model.compile(optimizer="adam", loss="mse", metrics=["mae"])
history = tf_model.fit(X_train, y_train, epochs=10, verbose=0, validation_split=0.2)

tf_model.save("models/price_predictor_tf.h5")
print("✅ TensorFlow Model Trained and Saved!\n")

# Visualization: Loss Curve
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("TensorFlow Model Training Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# --------------------------------------
# Step 7: Output Summary
# --------------------------------------

print("\n✅ All Tasks Completed Successfully!")
print("📂 Models saved in 'models/' folder")
print("📊 Dataset saved in 'data/' folder")
print("📈 Visualizations generated above\n")

print("--- Final Jewelry Dataset with Sentiments ---")
print(df.head())

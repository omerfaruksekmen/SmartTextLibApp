from google.colab import drive
drive.mount("/content/drive")

# Gerekli kütüphaneler
import pandas as pd
import numpy as np
import string
import pickle

# NLTK ön işleme
import nltk
nltk.download('all')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words and w.isalpha()]
    return ' '.join(tokens)

# CSV dosyalarını yükle
train_df = pd.read_csv("train.csv", header=0)
test_df = pd.read_csv("test.csv", header=0)

# Etiketleri 0-3 aralığına çek
train_df['Class Index'] = train_df['Class Index'].astype(int) - 1
test_df['Class Index'] = test_df['Class Index'].astype(int) - 1

# İsim değiştir
train_df.rename(columns={'Class Index': 'label', 'Title': 'title', 'Description': 'description'}, inplace=True)
test_df.rename(columns={'Class Index': 'label', 'Title': 'title', 'Description': 'description'}, inplace=True)

# Giriş metnini birleştir ve NLTK ile temizle
train_texts = (train_df['title'] + " " + train_df['description']).apply(preprocess).values
test_texts = (test_df['title'] + " " + test_df['description']).apply(preprocess).values
train_labels = train_df['label'].values
test_labels = test_df['label'].values

# TF-IDF vektörleştirme
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(train_texts)
X_test_tfidf = vectorizer.transform(test_texts)

# Makine Öğrenmesi Modeli (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

ml_model = LogisticRegression(max_iter=200)
ml_model.fit(X_train_tfidf, train_labels)

# Ağırlıklar
weights = ml_model.coef_
bias = ml_model.intercept_

weights_t = weights.T

import tensorflow as tf

# Sabitler
num_features = weights_t.shape[0]
num_classes = weights_t.shape[1]

# Keras modeli
model_tf = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(num_features,)),
    tf.keras.layers.Dense(num_classes, activation='softmax', use_bias=True)
])

# Ağırlıkları set et
model_tf.layers[0].set_weights([weights_t, bias])

# Modeli test et
dummy_input = np.zeros((1, num_features), dtype=np.float32)
deneme_preds = model_tf.predict(dummy_input)
print("Test prediction:", deneme_preds)

converter = tf.lite.TFLiteConverter.from_keras_model(model_tf)
tflite_model = converter.convert()

with open("logistic_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model TFLite olarak kaydedildi.")

import pickle
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(ml_model, f)

from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    estimator=ml_model,
    X=X_train_tfidf,
    y=train_labels,
    cv=5,
    scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Accuracy')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation Accuracy')
plt.title("Logistic Regression - Learning Curve")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score

# ML değerlendirme
y_pred = ml_model.predict(X_test_tfidf)

# Confusion Matrix
cm_lr = confusion_matrix(test_labels, y_pred)
disp_lr = ConfusionMatrixDisplay(confusion_matrix=cm_lr, display_labels=["World", "Sports", "Business", "Sci/Tech"])
disp_lr.plot(cmap=plt.cm.Blues)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title("Logistic Regression - Confusion Matrix")
plt.show()

# Sınıflandırma Raporu
print("\nClassification Report (Logistic Regression):")
print(classification_report(test_labels, y_pred, target_names=["World", "Sports", "Business", "Sci/Tech"]))

# Örnek custom metin
custom_text = "Shares of the electric vehicle manufacturer surged after the company reported record-breaking quarterly earnings and announced plans for expanding its production facilities across Asia."

# Ön işleme (NLTK temizleme)
custom_text_clean = preprocess(custom_text)

# TF-IDF ile vektörleştir
custom_vec = vectorizer.transform([custom_text_clean])

# Tahmin yap
custom_pred = ml_model.predict(custom_vec)[0]

# Etiket isimleri
label_names = ["World", "Sports", "Business", "Sci/Tech"]

print(f"Custom Metin: {custom_text}")
print(f"Tahmin Edilen Sınıf: {label_names[custom_pred]}")

# Derin Öğrenme Modeli
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

y_train_cat = to_categorical(train_labels, num_classes=4)
y_test_cat = to_categorical(test_labels, num_classes=4)

dl_model = Sequential([
    Dense(512, activation='relu', input_shape=(5000,)),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.2),
    Dense(4, activation='softmax')
])

dl_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

dl_model.fit(X_train_tfidf.toarray(), y_train_cat, validation_data=(X_test_tfidf.toarray(), y_test_cat),
             epochs=5, batch_size=32)

# Örnek bir cümle - custom veri ile test
text = "The European Union has announced new trade sanctions in response to human rights violations. Diplomats say the measures aim to pressure the foreign government without harming civilians. Talks are underway to determine the long-term impact of these restrictions. The move has drawn mixed reactions from global leaders."

# TF-IDF ile dönüştür
text_tfidf = vectorizer.transform([text])

# Tahmin yap
prediction = dl_model.predict(text_tfidf.toarray())
predicted_class = np.argmax(prediction)

# Etiket isimlerini getir
label_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}

# Sonuç
print(f"Predicted Class: {label_map[predicted_class]}")
print(f"Confidence: {np.max(prediction)*100:.2f}%")

# Değerlendirme
loss, acc = dl_model.evaluate(X_test_tfidf.toarray(), y_test_cat)
print(f"\nDeep Learning Model Validation Accuracy: {acc:.2f}")

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# Test verisi ile tahmin
y_pred_dl_probs = dl_model.predict(X_test_tfidf.toarray())
y_pred_dl = np.argmax(y_pred_dl_probs, axis=1)

# Confusion Matrix
cm_dl = confusion_matrix(test_labels, y_pred_dl)
disp_dl = ConfusionMatrixDisplay(confusion_matrix=cm_dl, display_labels=["World", "Sports", "Business", "Sci/Tech"])
disp_dl.plot(cmap=plt.cm.Blues)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.title("Deep Learning (FNN) - Confusion Matrix")
plt.show()

# Sınıflandırma Raporu
print("Deep Learning - Classification Report")
print(classification_report(test_labels, y_pred_dl, target_names=["World", "Sports", "Business", "Sci/Tech"]))

# Makine öğrenmesi modeli doğruluğu
ml_accuracy = ml_model.score(X_test_tfidf, test_labels)

# Derin öğrenme modeli doğruluğu
loss_dl, acc_dl = dl_model.evaluate(X_test_tfidf.toarray(), y_test_cat, verbose=0)

models = ['Logistic Regression', 'Feedforward NN']
accuracies = [ml_accuracy, acc_dl]

plt.figure(figsize=(8,5))
bars = plt.bar(models, accuracies, color=['skyblue', 'salmon'])
plt.ylim([0, 1])
plt.ylabel('Accuracy')
plt.title('Accuracy Karşılaştırması: Makine Öğrenmesi vs Derin Öğrenme')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height - 0.05, f'{height:.3f}', ha='center', color='black', fontsize=12)

plt.show()

# Model ve vektörizer kaydetme
dl_model.save("ag_news_deep_model.h5")
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

import tensorflow as tf

# Keras modeli yüklenmesi
# model = tf.keras.models.load_model("ag_news_deep_model.h5")

# TensorFlow Lite Converter'ı oluşturulması
converter = tf.lite.TFLiteConverter.from_keras_model(dl_model)

# Modelin dönüştürülmesi
tflite_model = converter.convert()

# .tflite dosyasının kaydedilmesi
with open("ag_news_deep_model.tflite", "wb") as f:
    f.write(tflite_model)

import json
vocab = vectorizer.get_feature_names_out()
with open("tfidf_vocab.json", "w") as f:
    json.dump(vocab.tolist(), f)

import json

vocab = vectorizer.vocabulary_

idf = vectorizer.idf_

vocab_list = [None] * len(vocab)
for word, index in vocab.items():
    vocab_list[index] = word

with open("tfidf_full.json", "w") as f:
    json.dump({
        "vocab": vocab_list,
        "idf": idf.tolist()
    }, f)

!pip install lime --quiet

from lime.lime_text import LimeTextExplainer
import numpy as np

class_names = ["World", "Sports", "Business", "Sci/Tech"]

# Tahminleri olasılık olarak döndürme
def predict_proba(texts):
    vecs = vectorizer.transform(texts)
    return ml_model.predict_proba(vecs)

explainer = LimeTextExplainer(class_names=class_names)

sample_text = "NASA's new space telescope has sent back its first images of distant galaxies, revealing stunning views of the early universe."

# Açıklama oluşturulması
exp = explainer.explain_instance(sample_text, predict_proba, num_features=10, top_labels=1)

# Görselleştirme
exp.show_in_notebook(text=sample_text)

# HTML çıktı kaydet
exp.save_to_file("lime_output.html")

# LIME
def predict_proba_dl(texts):
    vecs = vectorizer.transform(texts)
    return dl_model.predict(vecs.toarray())

explainer = LimeTextExplainer(class_names=class_names)

sample_text = "I wake up early every day and run for half an hour at a light pace. Then I use dumbbells to do weight training."

# Açıklama oluşturulması
exp = explainer.explain_instance(sample_text, predict_proba_dl, num_features=10, top_labels=1)

# Görselleştirme
exp.show_in_notebook(text=sample_text)

# HTML çıktı kaydet
exp.save_to_file("dl_lime_output.html")
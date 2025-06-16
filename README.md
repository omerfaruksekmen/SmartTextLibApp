<h1 align="center">:iphone::newspaper:🔍📈SmartTextLibApp - Mobile Text Categorization & Explainable NLP App</h1>

<p align="center">
  <a href="https://github.com/omerfaruksekmen"><img src="https://img.shields.io/badge/GitHub-omerfaruksekmen-4c00e6?style=for-the-badge&logo=github&logoColor=white" alt="GitHub Badge"></a>
  <img src="https://img.shields.io/badge/API-24%2B-green?style=for-the-badge" alt="Kotlin Badge">
  <img src="https://img.shields.io/badge/KOTLIN-blue?style=for-the-badge&logo=kotlin&logoColor=purple&labelColor=orange" alt="Kotlin Badge">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python Badge">
  <img src="https://img.shields.io/badge/google_colab-F9AB00?style=for-the-badge&logo=google-colab&logoColor=white" alt="Google Colab Badge">
</p>

<p align="center">
  SmartTextLib is an AI-powered mobile application designed for text classification, specifically targeting news articles. 
  The app simultaneously utilizes both deep learning and machine learning models to categorize a given text into one of four predefined categories: World, Sports, Business, and Sci/Tech.
  What sets SmartTextLib apart is its dual functionality:
</p>

- It serves as an intelligent text categorizer for digital library or media applications.
- It acts as an educational NLP tool, offering users the ability to explore and understand key natural language processing (NLP) steps such as tokenization, TF-IDF vectorization, model inference, and explanation.

<p align="center">
  Thanks to the integration of explainable AI (XAI), the app provides detailed insights into how predictions are made, enhancing transparency and user trust. 
  The models are pre-trained and optimized for mobile use via TensorFlow Lite, ensuring fast and efficient on-device performance.
</p>

## :camera_flash: Screenshots

<table style="width: 100%; table-layout: fixed;">
  <tr>
    <th style="text-align: center; border: none;">Home Page</th>
    <th style="text-align: center; border: none;">Classification Page</th>
    <th style="text-align: center; border: none;">Text Entry - Prediction Result</th>
  </tr>
  <tr>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/1.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/2.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/3.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">Explain Button</th>
    <th style="text-align: center; border: none;">Save Document</th>
    <th style="text-align: center; border: none;">Category Page</th>
  </tr>
  <tr>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/4.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/5.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/6.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">Delete Document</th>
    <th style="text-align: center; border: none;">Empty Category</th>
    <th style="text-align: center; border: none;">Explainable AI Page - 1</th>
  </tr>
  <tr>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/7.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/8.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/9.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
  </tr>
  <tr>
    <th style="text-align: center; border: none;">Explainable AI Page - 2</th>
  </tr>
  <tr>
    <td style="height: 300px; width: 33.33%; text-align: center; border: none;">
      <img src="screenshots/10.png" style="width: 100%; height: 100%; object-fit: cover;" />
    </td>
  </tr>
</table>

## :hammer_and_wrench: Technologies
- Android
- Minimum SDK Level 24
- Kotlin
- Python
- NLTK (Natural Language Toolkit)
- Scikit-learn
- Tensorflow / Keras
- Tensorflow Lite (TFLite)
- AG News Dataset
- Explainable AI (XAI)

## How To Run 🚀

1. 📱 Requirements
- Android device (Android 8.0 or higher recommended)
- Android Studio (for building from source)
- Internet connection (only required for downloading the APK or updating resources)
2. 🧠 Using the App
- On the main screen, enter a news headline or text content.
- Tap the "Predict Category" button.
- View classification results from both Machine Learning (Logistic Regression) and Deep Learning models.
- Tap "Explain" to explore the NLP pipeline and explanation for the prediction.
- Use "Educational Mode" to view step-by-step processing like tokenization, TF-IDF, top keywords, and prediction confidence.
3. 🎯 Categories
The app classifies news into the following four categories:
- World
- Sports
- Business
- Sci/Tech

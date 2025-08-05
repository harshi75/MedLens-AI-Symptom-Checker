# MedLens – AI-Powered Symptom Checker 🩺

MedLens is an intelligent health assistant that uses machine learning to predict possible diseases based on selected symptoms.

## 🔧 Features
- Symptom-based disease prediction using Naive Bayes
- Streamlit-powered chatbot UI
- Lightweight, offline compatible

## 🚀 How to Run
1. Install requirements:
```bash
pip install -r requirements.txt
```
2. Start the app:
```bash
streamlit run streamlit_app.py
```

## 📊 Dataset
A small symptom-disease mapping dataset is included. You can replace it with a more detailed one.

## 🧠 ML Model
- Algorithm: Multinomial Naive Bayes
- Accuracy: ~85% on sample data

## 📎 Files
- `main.py` - Training and model creation
- `streamlit_app.py` - Chatbot interface
- `model.pkl` - Trained model
- `dataset.csv` - Symptom–disease data

## 🙋‍♀️ Author
Harshita Singh – AIML Student, Nitra Technical Campus

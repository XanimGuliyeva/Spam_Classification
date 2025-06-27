# Email Spam Classification Project

A machine learning project that classifies emails as spam or ham (legitimate) using both traditional ML and deep learning approaches.

## Overview

This project implements two different approaches to email spam classification:
- **Traditional ML**: Naive Bayes with TF-IDF vectorization
- **Deep Learning**: DistilBERT transformer model

## Dataset

- **Source**: Enron Email Dataset from Kaggle
- **Structure**: Ham and spam emails across multiple folders (enron1-enron4)
- **Preprocessing**: Text cleaning, stop word removal, and tokenization

## Models Implemented

### 1. Naive Bayes Classifier
- Uses TF-IDF vectorization (max 5000 features)
- Traditional machine learning approach
- Fast training and prediction

### 2. DistilBERT Classifier
- Pre-trained transformer model fine-tuned for email classification
- State-of-the-art deep learning approach
- Higher accuracy but more computationally intensive

## Features

- **Data Visualization**: Distribution plots, word clouds, and frequency analysis
- **Model Evaluation**: Accuracy, precision, recall, F1-score, and confusion matrices
- **Training Monitoring**: Loss curves and performance tracking
- **Model Persistence**: Save trained models for future use

## Requirements

```
pandas
scikit-learn
transformers
torch
matplotlib
seaborn
wordcloud
nltk
kaggle
numpy
joblib
```

## Usage

1. **Setup Environment**:
   ```bash
   pip install kaggle transformers --quiet
   ```

2. **Download Dataset**:
   ```bash
   kaggle datasets download -d wanderfj/enron-spam
   ```

3. **Run the Classification**:
   - Execute the notebook/script to train both models
   - Models will be automatically saved after training

## Results

The project generates:
- Performance metrics for both models
- Visualization plots (word clouds, confusion matrices)
- Saved model files for deployment

## File Structure

```
├── enron_email_dataset/     # Dataset folder
├── results/                 # Training outputs
├── logs/                   # Training logs
├── naive_bayes_model.pkl   # Saved NB model
├── tfidf_vectorizer.pkl    # Saved vectorizer
└── bert_saved_model/       # Saved BERT model
```

## Key Insights

- Comparative analysis between traditional ML and transformer approaches
- Word frequency analysis reveals common spam indicators
- Visualization of model performance and training progress

## Future Improvements

- Experiment with other transformer models (BERT, RoBERTa)
- Implement ensemble methods
- Add real-time email classification API
- Extend to multilingual spam detection

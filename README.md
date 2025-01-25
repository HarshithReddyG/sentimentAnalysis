# Sentiment Analysis for Restaurant Reviews

This project performs sentiment analysis on restaurant reviews using a Naive Bayes classifier. The aim is to classify reviews as positive or negative based on their text content. The code handles data preprocessing, feature extraction, model training, and evaluation.

---

## Features

- **Sentiment Classification**: Predicts whether a restaurant review is positive or negative.
- **Preprocessing Pipeline**: Includes tokenization, stopword removal, and lemmatization.
- **Feature Extraction**: Uses Bag-of-Words (BoW) and Term Frequency-Inverse Document Frequency (TF-IDF).
- **Naive Bayes Classifier**: Implements a probabilistic model for classification.
- **Model Evaluation**: Provides accuracy, precision, recall, and F1-score metrics.

---

## Requirements

To run this code, the following libraries are required:

- Python (>=3.7)
- NumPy
- Pandas
- Scikit-learn
- NLTK

Install the dependencies using the following command:
```bash
pip install numpy pandas scikit-learn nltk
```

---

## File Structure

```
.
|-- sentimentAnalysis.py   # Main Python script for sentiment analysis
|-- data/
|   |-- reviews.csv        # Input dataset of restaurant reviews
|-- output/
    |-- results.txt        # Model evaluation metrics
```

---

## How to Use

1. **Prepare the Dataset**:
   - Place the dataset (CSV file) containing restaurant reviews in the `data/` folder.
   - Ensure the dataset has the following structure:
     ```
     review,text,label
     1,"The food was amazing!",positive
     2,"Terrible service and cold food.",negative
     ```

2. **Run the Script**:
   - Execute the script using the following command:
     ```bash
     python sentimentAnalysis.py
     ```

3. **Output**:
   - The script will output metrics to the console and save detailed results in the `output/results.txt` file.

---

## Implementation Details

### 1. Preprocessing
- Converts text to lowercase.
- Removes punctuation and non-alphanumeric characters.
- Tokenizes and lemmatizes text using the NLTK library.
- Removes stopwords to focus on meaningful words.

### 2. Feature Extraction
- **Bag-of-Words (BoW)**: Creates a sparse matrix of word counts.
- **TF-IDF**: Assigns importance to words based on frequency and document-level significance.

### 3. Naive Bayes Classifier
- Implements a multinomial Naive Bayes classifier using Scikit-learn.
- Trains the model on the preprocessed dataset.

### 4. Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score

---

## Example Output

```
Model Evaluation Results:
-------------------------
Accuracy: 87.5%
Precision: 85.2%
Recall: 88.1%
F1-Score: 86.6%
```

---

## Next Steps

To improve the model, consider:
- Using advanced feature extraction methods such as word embeddings (Word2Vec, GloVe).
- Implementing deep learning models like LSTMs, GRUs, or Transformers (e.g., BERT).
- Hyperparameter tuning to optimize the Naive Bayes classifier.
- Expanding the dataset to include more diverse reviews.

---



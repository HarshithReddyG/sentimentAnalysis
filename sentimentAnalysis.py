#importing libraries 
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import matplotlib.pyplot as plt # type: ignore

#importing data
dataset = pd.read_csv('./Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

#deep cleaning of the texts 
import re 
import nltk 
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
corpus = []

for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)

#creating bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

#Splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Training the Naive Bayes model on the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting the test set results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

#Visualizing the confusion matrix
import seaborn as sns
sns.heatmap(cm, annot = True)
plt.show()

#Calculating the accuracy
accuracy = (cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])
print("Accuracy:", accuracy)

#Calculating the precision
precision = cm[1][1] / (cm[1][0] + cm[1][1])
print("Precision:", precision)

#Calculating the recall
recall = cm[1][1] / (cm[0][1] + cm[1][1])
print("Recall:", recall)

#Calculating the F1 score
f1_score = 2 * (precision * recall) / (precision + recall)
print("F1 Score:", f1_score)

#Calculating the AUC-ROC score
from sklearn.metrics import roc_auc_score
auc_roc = roc_auc_score(y_test, y_pred)
print("AUC-ROC Score:", auc_roc)

#Calculating the average precision score
from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_test, y_pred)
print("Average Precision Score:", average_precision)

#predict a single instance
new_review = 'This restaurant was terrible. The service was slow and the food was bland.'
new_review_transformed = cv.transform([new_review]).toarray()
new_review_prediction = classifier.predict(new_review_transformed)
print("Prediction for new review:", new_review_prediction)


#training the SVM model on training set 
#supportVectorMachine Models
from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)
classifier.fit(X_train, y_train)
y_pred_svm = classifier.predict(X_test)

#Making the confusion matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)

#Visualizing the confusion matrix
sns.heatmap(cm_svm, annot = True)
plt.show()

#Calculating the accuracy
accuracy_svm = (cm_svm[0][0] + cm_svm[1][1]) / (cm_svm[0][0] + cm_svm[0][1] + cm_svm[1][0] + cm_svm[1][1])
print("Accuracy (SVM):", accuracy_svm)

#Calculating the precision
precision_svm = cm_svm[1][1] / (cm_svm[1][0] + cm_svm[1][1])
print("Precision (SVM):", precision_svm)

#Calculating the recall
recall_svm = cm_svm[1][1] / (cm_svm[0][1] + cm_svm[1][1])
print("Recall (SVM):", recall_svm)

#Calculating the F1 score
f1_score_svm = 2 * (precision_svm * recall_svm) / (precision_svm + recall_svm)
print("F1 Score (SVM):", f1_score_svm)

#Calculating the AUC-ROC score
from sklearn.metrics import roc_auc_score
auc_roc_svm = roc_auc_score(y_test, y_pred_svm)
print("AUC-ROC Score (SVM):", auc_roc_svm)

#Calculating the average precision score
from sklearn.metrics import average_precision_score
average_precision_svm = average_precision_score(y_test, y_pred_svm)
print("Average Precision Score (SVM):", average_precision_svm)


#logisticRegressionModel

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_train)

#predicting the test set results
y_pred_lr = classifier.predict(X_test)

#Making the confusion matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)

#Visualizing the confusion matrix
sns.heatmap(cm_lr, annot = True)
plt.show()

#Calculating the accuracy
accuracy_lr = (cm_lr[0][0] + cm_lr[1][1]) / (cm_lr[0][0] + cm_lr[0][1] + cm_lr[1][0] + cm_lr[1][1])
print("Accuracy (Logistic Regression):", accuracy_lr)

#Calculating the precision
precision_lr = cm_lr[1][1] / (cm_lr[1][0] + cm_lr[1][1])
print("Precision (Logistic Regression):", precision_lr)

#Calculating the recall
recall_lr = cm_lr[1][1] / (cm_lr[0][1] + cm_lr[1][1])
print("Recall (Logistic Regression):", recall_lr)

#Calculating the F1 score
f1_score_lr = 2 * (precision_lr * recall_lr) / (precision_lr + recall_lr)
print("F1 Score (Logistic Regression):", f1_score_lr)       

#Calculating the AUC-ROC score
from sklearn.metrics import roc_auc_score
auc_roc_lr = roc_auc_score(y_test, y_pred_lr)
print("AUC-ROC Score (Logistic Regression):", auc_roc_lr)

#Calculating the average precision score
from sklearn.metrics import average_precision_score
average_precision_lr = average_precision_score(y_test, y_pred_lr)
print("Average Precision Score (Logistic Regression):", average_precision_lr)



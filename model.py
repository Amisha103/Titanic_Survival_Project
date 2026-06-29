import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
df = sns.load_dataset('titanic')
print(df.head())
df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male'], axis=1, inplace=True)

print(df.info())
df['age'].fillna(df['age'].mean(), inplace=True)
df.dropna(subset=['embarked'], inplace=True)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df = df.astype(int)

X = df.drop('survived', axis=1)
y = df['survived']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#LOGISTIC REGRESSION

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{cm}')
print(f'Classification Report:\n{cr}')

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#KNN

from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train)
y_pred_knn = knn_model.predict(X_test_scaled)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)
print(f'KNN Accuracy: {accuracy_knn}')
print(f'KNN Confusion Matrix:\n{cm_knn}')
print(f'KNN Classification Report:\n{classification_report(y_test, y_pred_knn)}')

#NAIVE BAYES

from sklearn.naive_bayes import GaussianNB
model_NB = GaussianNB() 
model_NB.fit(X_train, y_train)
y_pred_NB = model_NB.predict(X_test)
print(y_pred_NB)
accuracy_NB = accuracy_score(y_test, y_pred_NB)
cm_NB = confusion_matrix(y_test, y_pred_NB)
print(f'Naive Bayes Accuracy: {accuracy_NB}')
print(f'Naive Bayes Confusion Matrix:\n{cm_NB}')
print(f'Naive Bayes Classification Report:\n{classification_report(y_test, y_pred_NB)}')



import pandas as pd 
import numpy as np
df = pd.read_csv('part3_credit_risk_dirty.csv')
df.describe()
df.info()
df.isnull().sum()
df.dropna(subset=["risk_level"], inplace=True)
df['risk_level'].value_counts()
# Handling invalid debt_ratio values
df[df["debt_ratio"] < 0]
df[df["debt_ratio"] > 1]
df.loc[(df["debt_ratio"] < 0) | (df["debt_ratio"] > 1), "debt_ratio"] = np.nan
df['debt_ratio'].fillna(df['debt_ratio'].median(), inplace=True)
# Handling invalid monthly_income values
df[df["monthly_income"] < 0]
df.loc[df["monthly_income"] < 0, "monthly_income"] = np.nan
df['monthly_income'].fillna(df['monthly_income'].median(), inplace=True)
# Handling invalid credit_history years values
df[df["credit_history_years"] < 0]
df.loc[df["credit_history_years"] < 0, "credit_history_years"] = np.nan
df['credit_history_years'].fillna(df['credit_history_years'].median(), inplace=True)
df.isnull().sum()
# handling invalid age values
df[df["age"] < 0]
df["age"].fillna(df["age"].median(), inplace=True)
# handling invalid job_stability_years values
df[df["job_stability_years"] < 0]
df.loc[df["job_stability_years"] < 0, "job_stability_years"] = np.nan
df['job_stability_years'].fillna(df['job_stability_years'].median(), inplace=True)
df.isnull().sum()
df.to_csv('part3_credit_risk_cleaned.csv', index=False)
df.describe()
# Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['risk_level'])
label_encoder.classes_
df['risk_level'] = y
df.to_csv('part3_credit_risk_encoded.csv', index=False)
df.head()
df["risk_level"].value_counts()

X = df.drop("risk_level", axis=1)
y = df["risk_level"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier 
clf = DecisionTreeClassifier (criterion="gini", 
    max_depth=5,     
    random_state=42, 
    min_samples_split=3, 
    min_samples_leaf= 2 ,
      )
clf.fit(X_train, y_train)
from sklearn.metrics import accuracy_score, classification_report

y_pred = clf.predict(X_test)
accuracy_score(y_test, y_pred)

# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     "criterion": ["gini", "entropy"],
#     "max_depth": [3, 5, 8, 10, None],
#     "min_samples_split": [2, 5, 10],
#     "min_samples_leaf": [1, 2, 5],
#     "class_weight": [None, "balanced"]
# }
# grid = GridSearchCV(
#     DecisionTreeClassifier(random_state=42),
#     param_grid,
#     cv=5,
#     scoring="f1_weighted",   # meilleur que accuracy si déséquilibre
#     n_jobs=-1
# )
# grid.fit(X_train, y_train)
# best_model = grid.best_estimator_
# y_pred = best_model.predict(X_test)
# print("Accuracy :", accuracy_score(y_test, y_pred))
# accuracy_score(y_test, y_pred)
# print(classification_report(y_test, y_pred))



from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt 
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

import pickle

with open("credit_risk_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)